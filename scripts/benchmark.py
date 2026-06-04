#!/usr/bin/env python3
"""
Benchmark: ES script_score (exact KNN) vs knn (HNSW ANN)

Metrics measured per scale (10k / 50k / 100k docs):
  - Latency  : p50 / p95 / p99 over N_QUERIES runs
  - Recall@K : |exact_ids ∩ ann_ids| / TOP_K  (requires exact run)
  - Throughput: QPS under CONCURRENCY concurrent clients

Usage:
    python scripts/benchmark.py                   # all scales, all methods
    python scripts/benchmark.py --scale 10k       # single scale
    python scripts/benchmark.py --no-exact        # skip exact (it's slow at 100k)
"""

import sys
import os
import argparse
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from elasticsearch import Elasticsearch
from rich.console import Console
from rich.table import Table
from config import (
    ES_URL,
    INDEX_NAME,
    DIMS,
    SCALES,
    N_QUERIES,
    TOP_K,
    NUM_CANDIDATES_VARIANTS,
    CONCURRENCY,
    org_id,
)

console = Console()


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------


def query_exact(org: str, vector: list) -> dict:
    """script_score brute-force exact KNN (current production method)."""
    return {
        "size": TOP_K,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"version": "v2"}},
                            {"term": {"organization_id": org}},
                        ]
                    }
                },
                "script": {
                    "source": "dotProduct(params.query_vector, 'embedding_vector_v2') + 1.0",
                    "params": {"query_vector": vector},
                },
            }
        },
        "_source": False,
    }


def query_hnsw(org: str, vector: list, num_candidates: int) -> dict:
    """Native knn HNSW ANN search."""
    return {
        "size": TOP_K,
        "knn": {
            "field": "embedding_vector_v2",
            "query_vector": vector,
            "k": TOP_K,
            "num_candidates": num_candidates,
            "filter": [
                {"term": {"version": "v2"}},
                {"term": {"organization_id": org}},
            ],
        },
        "_source": False,
    }


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def timed_search(es: Elasticsearch, body: dict) -> tuple[float, list[str]]:
    """Return (latency_ms, list_of_hit_ids).

    The ES 9 Python client accepts top-level query/knn/size as kwargs.
    We pass them via ** unpacking from the pre-built body dict.
    """
    t0 = time.perf_counter()
    resp = es.search(index=INDEX_NAME, **body)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    ids = [h["_id"] for h in resp["hits"]["hits"]]
    return elapsed_ms, ids


def percentile(data: list[float], p: int) -> float:
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(data_sorted) - 1)
    return data_sorted[lo] + (data_sorted[hi] - data_sorted[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def run_latency(
    es: Elasticsearch,
    org: str,
    query_vectors: list,
    method_name: str,
    body_fn,
) -> dict:
    """Run N_QUERIES queries and return latency stats (ms)."""
    latencies = []
    for vec in query_vectors:
        body = body_fn(org, vec)
        ms, _ = timed_search(es, body)
        latencies.append(ms)

    return {
        "method": method_name,
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "mean": statistics.mean(latencies),
    }


def run_recall(
    es: Elasticsearch,
    org: str,
    query_vectors: list,
    num_candidates: int,
) -> float:
    """Compute mean recall@TOP_K for ANN vs exact ground truth."""
    recalls = []
    for vec in query_vectors:
        _, exact_ids = timed_search(es, query_exact(org, vec))
        _, ann_ids = timed_search(es, query_hnsw(org, vec, num_candidates))
        if not exact_ids:
            continue
        overlap = len(set(exact_ids) & set(ann_ids))
        recalls.append(overlap / len(exact_ids))
    return statistics.mean(recalls) if recalls else 0.0


def run_throughput(
    es: Elasticsearch,
    org: str,
    query_vectors: list,
    method_name: str,
    body_fn,
    concurrency: int = CONCURRENCY,
) -> float:
    """Return QPS under `concurrency` concurrent clients."""
    bodies = [body_fn(org, vec) for vec in query_vectors]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(timed_search, es, b) for b in bodies]
        _ = [f.result() for f in as_completed(futures)]
    elapsed = time.perf_counter() - t0
    return len(bodies) / elapsed


# ---------------------------------------------------------------------------
# Per-scale orchestration
# ---------------------------------------------------------------------------


def benchmark_scale(es: Elasticsearch, scale: int, run_exact: bool) -> list[dict]:
    oid = org_id(scale)
    console.print(f"\n[bold cyan]Scale: {scale:,} docs  (org={oid})[/bold cyan]")

    # Verify data is loaded
    es.indices.refresh(index=INDEX_NAME)
    count = es.count(
        index=INDEX_NAME,
        query={"term": {"organization_id": oid}},
    )["count"]
    if count == 0:
        console.print(f"  [red]No data found for {oid}. Run 'make load' first.[/red]")
        return []
    console.print(f"  {count:,} docs indexed")

    # Generate query vectors (unit-normalized, same distribution as indexed data)
    rng = np.random.default_rng(42)
    qvecs_raw = rng.standard_normal((N_QUERIES, DIMS["v2"])).astype(np.float32)
    norms = np.linalg.norm(qvecs_raw, axis=1, keepdims=True)
    qvecs = (qvecs_raw / norms).tolist()

    # Warmup — fire N_WARMUP silent queries to heat the JVM and OS page cache
    # before any timed measurement.  Without this, the first method measured
    # absorbs JIT-compilation and GC costs, making sequential scales look flat.
    N_WARMUP = 5
    console.print(f"  Warming up ({N_WARMUP} queries)...")
    warmup_vec = qvecs[0]
    for _ in range(N_WARMUP):
        timed_search(es, query_hnsw(oid, warmup_vec, NUM_CANDIDATES_VARIANTS[0]))

    rows = []

    # ---- Exact KNN ----
    if run_exact:
        console.print("  Running exact KNN (script_score)...")
        lat = run_latency(
            es, oid, qvecs, "exact (script_score)", lambda o, v: query_exact(o, v)
        )
        tput = run_throughput(es, oid, qvecs, "exact", lambda o, v: query_exact(o, v))
        rows.append({**lat, "recall": "—", "qps": tput, "scale": scale})

    # ---- HNSW ANN variants ----
    for nc in NUM_CANDIDATES_VARIANTS:
        label = f"knn HNSW nc={nc}"
        console.print(f"  Running {label}...")
        lat = run_latency(
            es, oid, qvecs, label, lambda o, v, _nc=nc: query_hnsw(o, v, _nc)
        )
        tput = run_throughput(
            es, oid, qvecs, label, lambda o, v, _nc=nc: query_hnsw(o, v, _nc)
        )

        recall = None
        if run_exact:
            console.print(f"    Computing recall@{TOP_K} vs exact...")
            recall = run_recall(es, oid, qvecs, nc)

        rows.append(
            {
                **lat,
                "recall": f"{recall:.3f}" if recall is not None else "—",
                "qps": tput,
                "scale": scale,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(all_rows: list[dict]) -> None:
    t = Table(
        title="Benchmark Results — raijin_search_indexer (v2, 4096-dim)",
        show_lines=True,
    )
    t.add_column("Scale", justify="right")
    t.add_column("Method", style="bold")
    t.add_column("p50 (ms)", justify="right")
    t.add_column("p95 (ms)", justify="right")
    t.add_column("p99 (ms)", justify="right")
    t.add_column(f"Recall@{TOP_K}", justify="right")
    t.add_column("QPS", justify="right")

    for r in all_rows:
        t.add_row(
            f"{r['scale']:,}",
            r["method"],
            f"{r['p50']:.1f}",
            f"{r['p95']:.1f}",
            f"{r['p99']:.1f}",
            str(r["recall"]),
            f"{r['qps']:.1f}",
        )

    console.print(t)
    console.print(
        f"\n[dim]Settings: {N_QUERIES} queries/method, top-{TOP_K}, "
        f"{CONCURRENCY} concurrent clients for QPS[/dim]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["10k", "50k", "100k", "200k", "300k", "500k", "1m", "all"], default="all")
    p.add_argument(
        "--no-exact",
        action="store_true",
        help="Skip exact KNN (script_score) for all scales",
    )
    p.add_argument(
        "--no-exact-above",
        type=int,
        default=None,
        metavar="N",
        help="Skip exact KNN for scales larger than N docs (e.g. 500000)",
    )
    p.add_argument(
        "--json",
        metavar="FILE",
        help="Save raw results to a JSON file (e.g. results/es_results.json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    scale_map = {
        "10k":  [10_000],
        "50k":  [50_000],
        "100k": [100_000],
        "200k": [200_000],
        "300k": [300_000],
        "500k": [500_000],
        "all":  SCALES,
    }
    targets = scale_map[args.scale]

    es = Elasticsearch(ES_URL, request_timeout=120)
    if not es.ping():
        console.print(f"[red]ERROR: Cannot reach Elasticsearch at {ES_URL}[/red]")
        sys.exit(1)

    if args.no_exact:
        console.print("[yellow]Note: --no-exact set. Recall@K will not be computed.[/yellow]")
    if args.no_exact_above:
        console.print(
            f"[yellow]Note: exact KNN skipped for scales > {args.no_exact_above:,} docs.[/yellow]"
        )

    all_rows = []
    for scale in targets:
        run_exact = (
            not args.no_exact
            and (args.no_exact_above is None or scale <= args.no_exact_above)
        )
        rows = benchmark_scale(es, scale, run_exact)
        all_rows.extend(rows)

    if all_rows:
        console.print()
        print_table(all_rows)
        if args.json:
            import json, pathlib

            out = pathlib.Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(all_rows, f, indent=2)
            console.print(f"[dim]Results saved → {out}[/dim]")
    else:
        console.print("[yellow]No results. Check that data is loaded.[/yellow]")


if __name__ == "__main__":
    main()
