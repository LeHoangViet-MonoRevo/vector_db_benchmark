#!/usr/bin/env python3
"""
Benchmark Milvus HNSW ANN (+ optional FLAT exact) search.

Mirrors benchmark.py exactly — same metrics, same scales, same query vectors:
  - Latency  : p50 / p95 / p99 over N_QUERIES runs
  - Recall@K : |flat_ids ∩ hnsw_ids| / TOP_K  (requires FLAT collections)
  - Throughput: QPS under CONCURRENCY concurrent clients

HNSW params match ES: m=16, efConstruction=100
ef variants match ES num_candidates: 200, 500

Usage:
    python scripts/milvus_benchmark.py                   # all scales, all methods
    python scripts/milvus_benchmark.py --scale 10k
    python scripts/milvus_benchmark.py --no-exact        # skip FLAT exact + recall
"""

import sys
import os
import json
import argparse
import pathlib
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

GT_DIR = Path(__file__).parent.parent / "ground_truth"

import numpy as np
from pymilvus import connections, Collection
from rich.console import Console
from rich.table import Table
from config import (
    MILVUS_HOST,
    MILVUS_PORT,
    DIMS,
    SCALES,
    N_QUERIES,
    TOP_K,
    MILVUS_EF_VARIANTS,
    CONCURRENCY,
    org_id,
    milvus_collection_name,
)

console = Console()


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


def search_hnsw(col: Collection, vector: list, ef: int) -> tuple[float, list[str]]:
    t0 = time.perf_counter()
    results = col.search(
        data=[vector],
        anns_field="embedding_vector_v2",
        param={"metric_type": "IP", "params": {"ef": ef}},
        limit=TOP_K,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, [hit.id for hit in results[0]]


def search_flat(col: Collection, vector: list) -> tuple[float, list[str]]:
    t0 = time.perf_counter()
    results = col.search(
        data=[vector],
        anns_field="embedding_vector_v2",
        param={"metric_type": "IP", "params": {}},
        limit=TOP_K,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, [hit.id for hit in results[0]]


# ---------------------------------------------------------------------------
# Benchmark runners  (mirrors benchmark.py structure)
# ---------------------------------------------------------------------------


def percentile(data: list[float], p: int) -> float:
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(data_sorted) - 1)
    return data_sorted[lo] + (data_sorted[hi] - data_sorted[lo]) * (k - lo)


def run_latency(query_vectors: list, method_name: str, search_fn) -> dict:
    latencies = [search_fn(vec)[0] for vec in query_vectors]
    return {
        "method": method_name,
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "mean": statistics.mean(latencies),
    }


def gt_path(scale: int) -> Path:
    return GT_DIR / f"gt_{scale // 1000}k.json"


def load_or_compute_gt(flat_col: Collection, qvecs: list, scale: int, recompute: bool = False) -> list:
    path = gt_path(scale)
    if path.exists() and not recompute:
        console.print(f"  [dim]Ground truth: loading cache from {path.name}[/dim]")
        with open(path) as f:
            return json.load(f)

    console.print(f"  Computing ground truth via FLAT ({len(qvecs)} queries)...")
    gt = []
    for vec in qvecs:
        _, ids = search_flat(flat_col, vec)
        gt.append(ids)

    GT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(gt, f)
    console.print(f"  [green]Ground truth saved → {path.name}[/green]")
    return gt


def run_recall(hnsw_col: Collection, qvecs: list, gt: list, ef: int) -> float:
    recalls = []
    for vec, exact_ids in zip(qvecs, gt):
        _, ann_ids = search_hnsw(hnsw_col, vec, ef)
        if not exact_ids:
            continue
        recalls.append(len(set(exact_ids) & set(ann_ids)) / len(exact_ids))
    return statistics.mean(recalls) if recalls else 0.0


def run_throughput(
    query_vectors: list, search_fn, concurrency: int = CONCURRENCY
) -> float:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(search_fn, vec) for vec in query_vectors]
        _ = [f.result() for f in as_completed(futures)]
    return len(query_vectors) / (time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Per-scale orchestration
# ---------------------------------------------------------------------------


def benchmark_scale(scale: int, run_exact: bool, recompute_gt: bool = False) -> list[dict]:
    oid = org_id(scale)
    console.print(f"\n[bold cyan]Scale: {scale:,} docs  (org={oid})[/bold cyan]")

    rng = np.random.default_rng(42)
    qvecs_raw = rng.standard_normal((N_QUERIES, DIMS["v2"])).astype(np.float32)
    qvecs = (qvecs_raw / np.linalg.norm(qvecs_raw, axis=1, keepdims=True)).tolist()

    rows = []
    gt = None

    if run_exact:
        flat_col = Collection(milvus_collection_name(scale, flat=True))
        flat_col.load()

        gt = load_or_compute_gt(flat_col, qvecs, scale, recompute=recompute_gt)

        console.print("  Running exact (FLAT)...")
        lat = run_latency(qvecs, "exact (FLAT)", lambda v: search_flat(flat_col, v))
        tput = run_throughput(qvecs, lambda v: search_flat(flat_col, v))
        rows.append({**lat, "recall": "—", "qps": tput, "scale": scale})

        flat_col.release()

    hnsw_col = Collection(milvus_collection_name(scale))
    hnsw_col.load()
    console.print(f"  {hnsw_col.num_entities:,} docs in HNSW collection")

    N_WARMUP = 5
    console.print(f"  Warming up ({N_WARMUP} queries)...")
    for _ in range(N_WARMUP):
        search_hnsw(hnsw_col, qvecs[0], MILVUS_EF_VARIANTS[0])

    for ef in MILVUS_EF_VARIANTS:
        label = f"HNSW ef={ef}"
        console.print(f"  Running {label}...")
        lat = run_latency(qvecs, label, lambda v, _ef=ef: search_hnsw(hnsw_col, v, _ef))
        tput = run_throughput(qvecs, lambda v, _ef=ef: search_hnsw(hnsw_col, v, _ef))

        recall = None
        if gt is not None:
            recall = run_recall(hnsw_col, qvecs, gt, ef)

        rows.append(
            {
                **lat,
                "recall": f"{recall:.3f}" if recall is not None else "—",
                "qps": tput,
                "scale": scale,
            }
        )

    hnsw_col.release()
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(all_rows: list[dict]) -> None:
    t = Table(
        title=f"Milvus Benchmark — raijin_milvus (v2, {DIMS['v2']}-dim, HNSW m=16 efC=100)",
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
        "--no-exact", action="store_true", help="Skip FLAT exact search and recall"
    )
    p.add_argument(
        "--no-exact-above", type=int, default=None, metavar="N",
        help="Skip FLAT exact search for scales > N docs (e.g. 100000)",
    )
    p.add_argument(
        "--recompute-gt", action="store_true",
        help="Recompute and overwrite cached ground truth files",
    )
    p.add_argument(
        "--json",
        metavar="FILE",
        help="Save raw results to a JSON file (e.g. results/milvus_results.json)",
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
        "1m":   [1_000_000],
        "all":  SCALES,
    }
    targets = scale_map[args.scale]

    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    console.print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

    out = pathlib.Path(args.json) if args.json else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(out.read_text()) if out.exists() else []
    else:
        existing = []

    all_rows = []
    for scale in targets:
        run_exact = (
            not args.no_exact
            and (args.no_exact_above is None or scale <= args.no_exact_above)
        )
        rows = benchmark_scale(scale, run_exact, recompute_gt=args.recompute_gt)
        all_rows.extend(rows)

        if out and rows:
            # merge: keep existing rows for scales not in this run, replace for current scale
            merged = [r for r in existing if r["scale"] not in targets] + all_rows
            with open(out, "w") as f:
                json.dump(merged, f, indent=2)
            console.print(f"  [dim]Results saved → {out}[/dim]")

    if all_rows:
        console.print()
        print_table(all_rows)
    else:
        console.print("[yellow]No results. Check that data is loaded.[/yellow]")


if __name__ == "__main__":
    main()
