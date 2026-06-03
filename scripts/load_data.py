#!/usr/bin/env python3
"""
Load synthetic v2 (4096-dim) documents into raijin_search_indexer.

Generates unit-normalized random vectors — required for dot_product similarity
in HNSW. Each scale gets its own organization_id so benchmarks are isolated.

Usage:
    python scripts/load_data.py              # load all scales (10k, 50k, 100k)
    python scripts/load_data.py --scale 10k  # load one scale
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
from config import ES_URL, INDEX_NAME, DIMS, SCALES, BULK_BATCH, org_id


def unit_vectors(n: int, dims: int) -> np.ndarray:
    """Generate n unit-normalized random vectors of given dimensionality."""
    vecs = np.random.randn(n, dims).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def doc_iter(scale: int, vectors: np.ndarray):
    oid = org_id(scale)
    for i, vec in enumerate(vectors):
        yield {
            "_index": INDEX_NAME,
            "_id": f"{oid}_prod_{i:07d}",
            "_source": {
                "organization_id": oid,
                "product_id": f"prod_{i:07d}",
                "version": "v2",
                "embedding_vector_v2": vec.tolist(),
            },
        }


def load_scale(es: Elasticsearch, scale: int) -> None:
    oid = org_id(scale)

    # Check if already loaded
    es.indices.refresh(index=INDEX_NAME)
    count_resp = es.count(
        index=INDEX_NAME,
        query={"term": {"organization_id": oid}},
    )
    existing = count_resp["count"]
    if existing >= scale:
        print(f"  {oid}: {existing:,} docs already present — skipping")
        return

    print(f"  Generating {scale:,} unit-normalized vectors ({DIMS['v2']}d)...")
    t0 = time.perf_counter()
    vectors = unit_vectors(scale, DIMS["v2"])
    gen_time = time.perf_counter() - t0
    print(f"  Generated in {gen_time:.1f}s  ({vectors.nbytes / 1e6:.0f} MB in memory)")

    print(f"  Indexing into '{INDEX_NAME}' (org={oid})...")
    t1 = time.perf_counter()

    docs = list(doc_iter(scale, vectors))
    batches = [docs[i : i + BULK_BATCH] for i in range(0, len(docs), BULK_BATCH)]

    success = 0
    errors = []
    for batch in tqdm(batches, desc=f"  Bulk ({BULK_BATCH}/batch)"):
        ok, errs = bulk(es, batch, raise_on_error=False)
        success += ok
        errors.extend(errs)

    idx_time = time.perf_counter() - t1
    rate = success / idx_time

    print(f"  Indexed {success:,} docs in {idx_time:.1f}s  ({rate:.0f} docs/s)")
    if errors:
        print(f"  WARNING: {len(errors)} errors — first: {errors[0]}")


def finalize(es: Elasticsearch) -> None:
    """Reset refresh interval and force merge for consistent benchmark conditions."""
    print("\nFinalizing index...")
    es.indices.put_settings(
        index=INDEX_NAME,
        settings={"index": {"refresh_interval": "1s"}},
    )
    es.indices.refresh(index=INDEX_NAME)

    resp = es.count(index=INDEX_NAME)
    print(f"Total docs in index: {resp['count']:,}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scale",
        choices=["10k", "50k", "100k", "all"],
        default="all",
        help="Which scale(s) to load",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    scale_map = {
        "10k": [10_000],
        "50k": [50_000],
        "100k": [100_000],
        "all": SCALES,
    }
    targets = scale_map[args.scale]

    es = Elasticsearch(ES_URL, request_timeout=120)
    if not es.ping():
        print(f"ERROR: Cannot reach Elasticsearch at {ES_URL}")
        sys.exit(1)

    if not es.indices.exists(index=INDEX_NAME):
        print(f"ERROR: Index '{INDEX_NAME}' not found. Run 'make setup' first.")
        sys.exit(1)

    print(f"Loading scales: {[org_id(s) for s in targets]}")
    for scale in targets:
        print(f"\n--- Scale: {scale:,} docs ---")
        load_scale(es, scale)

    finalize(es)


if __name__ == "__main__":
    main()
