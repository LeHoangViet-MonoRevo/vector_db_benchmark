#!/usr/bin/env python3
"""
Load synthetic v2 (4096-dim) documents into Milvus collections.

Generates unit-normalized random vectors — required for IP similarity to equal
cosine similarity, matching the ES index contract.

Usage:
    python scripts/milvus_load.py              # load all scales
    python scripts/milvus_load.py --scale 10k
    python scripts/milvus_load.py --no-flat    # skip FLAT collections
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from tqdm import tqdm
from pymilvus import connections, Collection, utility
from config import (
    MILVUS_HOST,
    MILVUS_PORT,
    DIMS,
    SCALES,
    BULK_BATCH,
    org_id,
    milvus_collection_name,
)


CHUNK_SIZE = 3_000  # vectors per gRPC insert — 4096-dim float32 = ~49 MB, under 64 MB server limit


def load_collection(scale: int, flat: bool = False) -> None:
    name = milvus_collection_name(scale, flat=flat)
    label = "FLAT" if flat else "HNSW"

    if not utility.has_collection(name):
        print(f"  '{name}' not found — run milvus_setup.py first")
        return

    col = Collection(name)
    col.flush()
    existing = col.num_entities
    if existing >= scale:
        print(f"  {name} ({label}): {existing:,} docs already present — skipping")
        return

    oid = org_id(scale)
    seed = 1000 + scale  # deterministic — HNSW and FLAT get identical vectors
    rng = np.random.default_rng(seed)

    n_chunks = (scale + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  Streaming {scale:,} vectors in chunks of {CHUNK_SIZE:,} "
          f"(peak RAM ≈ {CHUNK_SIZE * DIMS['v2'] * 4 / 1e6:.0f} MB)")

    t0 = time.perf_counter()
    for chunk_idx in tqdm(range(n_chunks), desc=f"  {name}"):
        chunk_start = chunk_idx * CHUNK_SIZE
        n = min(CHUNK_SIZE, scale - chunk_start)

        # Generate only this chunk — freed at end of iteration
        vecs = rng.standard_normal((n, DIMS["v2"])).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        ids         = [f"{oid}_prod_{chunk_start + j:07d}" for j in range(n)]
        org_ids     = [oid] * n
        product_ids = [f"prod_{chunk_start + j:07d}" for j in range(n)]

        col.insert([ids, org_ids, product_ids, vecs.tolist()])

    col.flush()
    elapsed = time.perf_counter() - t0
    print(f"  Indexed {col.num_entities:,} docs in {elapsed:.1f}s  ({scale / elapsed:.0f} docs/s)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["10k", "50k", "100k", "200k", "300k", "500k", "1m", "all"], default="all")
    p.add_argument(
        "--no-flat", action="store_true", help="Skip FLAT (exact) collections"
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
    print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

    for scale in targets:
        print(f"\n--- Scale: {scale:,} docs ---")
        load_collection(scale, flat=False)
        if not args.no_flat:
            load_collection(scale, flat=True)


if __name__ == "__main__":
    main()
