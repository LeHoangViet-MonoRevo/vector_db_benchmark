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


def unit_vectors(n: int, dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dims)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


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
        col.load()
        return

    oid = org_id(scale)
    # Use a deterministic seed per scale so HNSW and FLAT collections get the same vectors
    seed = 1000 + scale
    print(
        f"  Generating {scale:,} unit-normalized vectors ({DIMS['v2']}d, seed={seed})..."
    )
    t0 = time.perf_counter()
    vectors = unit_vectors(scale, DIMS["v2"], seed)
    print(
        f"  Generated in {time.perf_counter() - t0:.1f}s  ({vectors.nbytes / 1e6:.0f} MB)"
    )

    ids = [f"{oid}_prod_{i:07d}" for i in range(scale)]
    org_ids = [oid] * scale
    product_ids = [f"prod_{i:07d}" for i in range(scale)]

    print(f"  Inserting into '{name}'...")
    t1 = time.perf_counter()
    for start in tqdm(range(0, scale, BULK_BATCH), desc=f"  Bulk ({BULK_BATCH}/batch)"):
        end = min(start + BULK_BATCH, scale)
        col.insert(
            [
                ids[start:end],
                org_ids[start:end],
                product_ids[start:end],
                vectors[start:end].tolist(),
            ]
        )

    col.flush()
    elapsed = time.perf_counter() - t1
    total = col.num_entities
    print(f"  Indexed {total:,} docs in {elapsed:.1f}s  ({scale / elapsed:.0f} docs/s)")

    print(f"  Loading '{name}' into memory...")
    col.load()
    print(f"  Ready.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["10k", "50k", "100k", "all"], default="all")
    p.add_argument(
        "--no-flat", action="store_true", help="Skip FLAT (exact) collections"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scale_map = {"10k": [10_000], "50k": [50_000], "100k": [100_000], "all": SCALES}
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
