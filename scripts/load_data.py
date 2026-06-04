#!/usr/bin/env python3
"""
Load synthetic v2 (4096-dim) documents into raijin_search_indexer.

Generates unit-normalized random vectors in chunks so peak RAM stays low
regardless of scale (chunk size controls the ceiling, not total docs).

Peak RAM = CHUNK_SIZE × 4096 × 4 bytes ≈ 164 MB at the default 10k chunk size,
versus ~8 GB for 500k if the whole array were generated at once.

Usage:
    python scripts/load_data.py              # load all scales
    python scripts/load_data.py --scale 500k
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

# Vectors are generated this many at a time.
# Peak RAM ≈ CHUNK_SIZE × 4096 × 4 bytes  (~164 MB at 10k)
CHUNK_SIZE = 10_000


def _chunk_docs(oid: str, chunk_start: int, vecs: np.ndarray):
    """Yield ES bulk-action dicts for one chunk of vectors."""
    for j, vec in enumerate(vecs):
        i = chunk_start + j
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

    es.indices.refresh(index=INDEX_NAME)
    existing = es.count(
        index=INDEX_NAME,
        query={"term": {"organization_id": oid}},
    )["count"]
    if existing >= scale:
        print(f"  {oid}: {existing:,} docs already present — skipping")
        return

    print(f"  Streaming {scale:,} vectors in chunks of {CHUNK_SIZE:,} "
          f"(peak RAM ≈ {CHUNK_SIZE * DIMS['v2'] * 4 / 1e6:.0f} MB)")

    rng = np.random.default_rng(42 + scale)  # deterministic per scale
    n_chunks = (scale + CHUNK_SIZE - 1) // CHUNK_SIZE
    success = 0
    errors = []
    t0 = time.perf_counter()

    for chunk_idx in tqdm(range(n_chunks), desc=f"  {oid}"):
        chunk_start = chunk_idx * CHUNK_SIZE
        n = min(CHUNK_SIZE, scale - chunk_start)

        # Generate only this chunk — freed at end of loop iteration
        vecs = rng.standard_normal((n, DIMS["v2"])).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        ok, errs = bulk(
            es,
            _chunk_docs(oid, chunk_start, vecs),
            chunk_size=BULK_BATCH,
            raise_on_error=False,
        )
        success += ok
        errors.extend(errs)

    elapsed = time.perf_counter() - t0
    print(f"  Indexed {success:,} docs in {elapsed:.1f}s  ({success / elapsed:.0f} docs/s)")
    if errors:
        print(f"  WARNING: {len(errors)} errors — first: {errors[0]}")


def finalize(es: Elasticsearch) -> None:
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
        choices=["10k", "50k", "100k", "200k", "300k", "500k", "1m", "all"],
        default="all",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        metavar="N",
        help=f"Vectors generated per chunk (default {CHUNK_SIZE:,}). "
             "Lower = less RAM, more round-trips.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global CHUNK_SIZE
    CHUNK_SIZE = args.chunk_size

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
