#!/usr/bin/env python3
"""
Benchmark ES exact KNN (script_score) vs HNSW ANN (knn).

Two modes:
  per-org (default) — one index per scale (raijin_bench_Xk), no org filter.
                      Clean HNSW graphs, best recall.
  shared  (--shared) — single shared index (raijin_search_indexer), org filter
                       at query time. Legacy approach, suffers recall degradation
                       at scale due to pre-filtering on mixed HNSW graph.

Metrics per scale:
  - Latency  : p50 / p95 / p99 over N_QUERIES runs
  - Recall@K : |exact_ids ∩ ann_ids| / TOP_K  (requires exact run)
  - Throughput: QPS under CONCURRENCY concurrent clients

Usage:
    python scripts/benchmark.py                            # per-org, all scales
    python scripts/benchmark.py --scale 300k
    python scripts/benchmark.py --no-exact
    python scripts/benchmark.py --no-exact-above 500000
    python scripts/benchmark.py --shared                   # legacy shared index
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
    ES_URL, INDEX_NAME, DIMS, SCALES, N_QUERIES, TOP_K,
    NUM_CANDIDATES_VARIANTS, CONCURRENCY, org_id, es_index_name,
)

console = Console()


# ---------------------------------------------------------------------------
# Query builders
# org=None  → per-org mode (index already isolated, no filter)
# org=str   → shared mode  (must filter by organization_id)
# ---------------------------------------------------------------------------

def query_exact(vector: list, org: str = None) -> dict:
    inner = (
        {"bool": {"filter": [{"term": {"organization_id": org}}, {"term": {"version": "v2"}}]}}
        if org else {"match_all": {}}
    )
    return {
        "size": TOP_K,
        "query": {
            "script_score": {
                "query": inner,
                "script": {
                    "source": "dotProduct(params.query_vector, 'embedding_vector_v2') + 1.0",
                    "params": {"query_vector": vector},
                },
            }
        },
        "_source": False,
    }


def query_hnsw(vector: list, num_candidates: int, org: str = None) -> dict:
    body = {
        "size": TOP_K,
        "knn": {
            "field": "embedding_vector_v2",
            "query_vector": vector,
            "k": TOP_K,
            "num_candidates": num_candidates,
        },
        "_source": False,
    }
    if org:
        body["knn"]["filter"] = [
            {"term": {"organization_id": org}},
            {"term": {"version": "v2"}},
        ]
    return body


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def timed_search(es: Elasticsearch, index: str, body: dict) -> tuple[float, list[str]]:
    t0 = time.perf_counter()
    resp = es.search(index=index, **body)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, [h["_id"] for h in resp["hits"]["hits"]]


def percentile(data: list[float], p: int) -> float:
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_latency(es, index, query_vectors, method_name, body_fn) -> dict:
    latencies = [timed_search(es, index, body_fn(v))[0] for v in query_vectors]
    return {
        "method": method_name,
        "p50":  percentile(latencies, 50),
        "p95":  percentile(latencies, 95),
        "p99":  percentile(latencies, 99),
        "mean": statistics.mean(latencies),
    }


def run_recall(es, index, query_vectors, num_candidates, org=None) -> float:
    recalls = []
    for v in query_vectors:
        _, exact_ids = timed_search(es, index, query_exact(v, org))
        _, ann_ids   = timed_search(es, index, query_hnsw(v, num_candidates, org))
        if exact_ids:
            recalls.append(len(set(exact_ids) & set(ann_ids)) / len(exact_ids))
    return statistics.mean(recalls) if recalls else 0.0


def run_throughput(es, index, query_vectors, body_fn, concurrency=CONCURRENCY) -> float:
    bodies = [body_fn(v) for v in query_vectors]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        _ = [f.result() for f in as_completed(
            ex.submit(timed_search, es, index, b) for b in bodies
        )]
    return len(bodies) / (time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Per-scale orchestration
# ---------------------------------------------------------------------------

def benchmark_scale(es: Elasticsearch, scale: int, run_exact: bool, shared: bool) -> list[dict]:
    oid   = org_id(scale) if shared else None
    index = INDEX_NAME if shared else es_index_name(scale)
    mode_label = f"shared · org={oid}" if shared else f"per-org · index={index}"
    console.print(f"\n[bold cyan]Scale: {scale:,} docs  ({mode_label})[/bold cyan]")

    if not es.indices.exists(index=index):
        console.print(f"  [red]Index '{index}' not found.[/red]")
        return []

    es.indices.refresh(index=index)
    count = (
        es.count(index=index, query={"term": {"organization_id": oid}})["count"]
        if shared else es.count(index=index)["count"]
    )
    if count == 0:
        console.print(f"  [red]No docs found. Load data first.[/red]")
        return []
    console.print(f"  {count:,} docs indexed")

    rng = np.random.default_rng(42)
    qvecs_raw = rng.standard_normal((N_QUERIES, DIMS["v2"])).astype(np.float32)
    qvecs = (qvecs_raw / np.linalg.norm(qvecs_raw, axis=1, keepdims=True)).tolist()

    N_WARMUP = 5
    console.print(f"  Warming up ({N_WARMUP} queries)...")
    for _ in range(N_WARMUP):
        timed_search(es, index, query_hnsw(qvecs[0], NUM_CANDIDATES_VARIANTS[0], oid))

    rows = []

    if run_exact:
        console.print("  Running exact KNN (script_score)...")
        lat  = run_latency(es, index, qvecs, "exact (script_score)", lambda v: query_exact(v, oid))
        tput = run_throughput(es, index, qvecs, lambda v: query_exact(v, oid))
        rows.append({**lat, "recall": "—", "qps": tput, "scale": scale})

    for nc in NUM_CANDIDATES_VARIANTS:
        label = f"knn HNSW nc={nc}"
        console.print(f"  Running {label}...")
        lat  = run_latency(es, index, qvecs, label, lambda v, _nc=nc: query_hnsw(v, _nc, oid))
        tput = run_throughput(es, index, qvecs, lambda v, _nc=nc: query_hnsw(v, _nc, oid))

        recall = None
        if run_exact:
            console.print(f"    Computing recall@{TOP_K} vs exact...")
            recall = run_recall(es, index, qvecs, nc, oid)

        rows.append({
            **lat,
            "recall": f"{recall:.3f}" if recall is not None else "—",
            "qps": tput,
            "scale": scale,
        })

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(all_rows: list, shared: bool) -> None:
    mode = "shared index" if shared else "per-org indices"
    t = Table(
        title=f"ES Benchmark ({mode}) — v2 {DIMS['v2']}-dim  HNSW m=16 efC=256",
        show_lines=True,
    )
    for col, kw in [
        ("Scale", {"justify": "right"}),
        ("Method", {"style": "bold"}),
        ("p50 (ms)", {"justify": "right"}),
        ("p95 (ms)", {"justify": "right"}),
        ("p99 (ms)", {"justify": "right"}),
        (f"Recall@{TOP_K}", {"justify": "right"}),
        ("QPS", {"justify": "right"}),
    ]:
        t.add_column(col, **kw)

    for r in all_rows:
        t.add_row(
            f"{r['scale']:,}", r["method"],
            f"{r['p50']:.1f}", f"{r['p95']:.1f}", f"{r['p99']:.1f}",
            str(r["recall"]), f"{r['qps']:.1f}",
        )
    console.print(t)
    console.print(
        f"\n[dim]{N_QUERIES} queries/method · top-{TOP_K} · {CONCURRENCY} concurrent clients[/dim]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scale",
                   choices=["10k", "50k", "100k", "200k", "300k", "500k", "1m", "all"],
                   default="all")
    p.add_argument("--no-exact", action="store_true",
                   help="Skip exact KNN for all scales")
    p.add_argument("--no-exact-above", type=int, default=None, metavar="N",
                   help="Skip exact KNN for scales > N docs (e.g. 500000)")
    p.add_argument("--shared", action="store_true",
                   help="Use shared index with org filter (legacy mode)")
    p.add_argument("--json", metavar="FILE", help="Save results to JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    scale_map = {
        "10k": [10_000], "50k": [50_000], "100k": [100_000],
        "200k": [200_000], "300k": [300_000], "500k": [500_000],
        "1m": [1_000_000], "all": SCALES,
    }
    targets = scale_map[args.scale]

    es = Elasticsearch(ES_URL, request_timeout=120)
    if not es.ping():
        console.print(f"[red]ERROR: Cannot reach Elasticsearch at {ES_URL}[/red]")
        sys.exit(1)

    console.print(
        "[yellow]Mode: shared index (org filter)[/yellow]" if args.shared
        else "[green]Mode: per-org indices[/green]"
    )

    all_rows = []
    for scale in targets:
        run_exact = (
            not args.no_exact
            and (args.no_exact_above is None or scale <= args.no_exact_above)
        )
        rows = benchmark_scale(es, scale, run_exact, args.shared)
        all_rows.extend(rows)

    if all_rows:
        console.print()
        print_table(all_rows, args.shared)
        if args.json:
            import json, pathlib
            out = pathlib.Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(all_rows, f, indent=2)
            console.print(f"[dim]Results saved → {out}[/dim]")
    else:
        console.print("[yellow]No results.[/yellow]")


if __name__ == "__main__":
    main()
