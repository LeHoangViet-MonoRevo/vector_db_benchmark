#!/usr/bin/env python3
"""
Create Milvus collections for benchmarking.

Matches ES index spec:
  - HNSW collections: m=16, efConstruction=100, IP metric (dot product for unit vecs)
  - FLAT collections:  brute-force exact search, used as recall ground truth

Usage:
    python scripts/milvus_setup.py              # create all collections
    python scripts/milvus_setup.py --delete     # drop and recreate
    python scripts/milvus_setup.py --no-flat    # HNSW only (skip exact collections)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from config import MILVUS_HOST, MILVUS_PORT, DIMS, SCALES, milvus_collection_name


HNSW_INDEX = {
    "metric_type": "IP",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 100},
}

FLAT_INDEX = {
    "metric_type": "IP",
    "index_type": "FLAT",
    "params": {},
}


def make_schema() -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema("doc_id", DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema("organization_id", DataType.VARCHAR, max_length=50),
            FieldSchema("product_id", DataType.VARCHAR, max_length=50),
            FieldSchema("embedding_vector_v2", DataType.FLOAT_VECTOR, dim=DIMS["v2"]),
        ],
        description="raijin benchmark collection",
    )


def create_collection(name: str, index_params: dict, delete: bool) -> None:
    if utility.has_collection(name):
        if delete:
            utility.drop_collection(name)
            print(f"  Dropped '{name}'")
        else:
            print(f"  '{name}' already exists — skipping (use --delete to recreate)")
            return

    col = Collection(name=name, schema=make_schema())
    col.create_index(field_name="embedding_vector_v2", index_params=index_params)
    idx_type = index_params["index_type"]
    params = index_params.get("params", {})
    print(f"  Created '{name}' — {idx_type} {params}  metric=IP  dims={DIMS['v2']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--delete", action="store_true", help="Drop and recreate existing collections"
    )
    p.add_argument(
        "--no-flat", action="store_true", help="Skip FLAT (exact) collections"
    )
    args = p.parse_args()

    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

    for scale in SCALES:
        print(f"\nScale {scale // 1000}k:")
        create_collection(milvus_collection_name(scale), HNSW_INDEX, args.delete)
        if not args.no_flat:
            create_collection(
                milvus_collection_name(scale, flat=True), FLAT_INDEX, args.delete
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
