#!/usr/bin/env python3
"""Create raijin_search_indexer with dense_vector fields for v2/v3/3d embeddings."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from elasticsearch import Elasticsearch
from config import ES_URL, INDEX_NAME, DIMS


MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        # Slow refresh during bulk loading; scripts/load_data.py resets to 1s after load
        "refresh_interval": "30s",
    },
    "mappings": {
        "properties": {
            "organization_id": {"type": "keyword"},
            "product_id": {"type": "keyword"},
            "version": {"type": "keyword"},
            # dot_product requires unit-normalized vectors (same ranking as cosine for
            # unit vecs, but avoids normalization overhead at query time)
            "embedding_vector_v2": {
                "type": "dense_vector",
                "dims": DIMS["v2"],
                "index": True,
                "similarity": "dot_product",
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 256},
            },
            "embedding_vector_v3": {
                "type": "dense_vector",
                "dims": DIMS["v3"],
                "index": True,
                "similarity": "dot_product",
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 256},
            },
            "embedding_vector_3d": {
                "type": "dense_vector",
                "dims": DIMS["3d"],
                "index": True,
                "similarity": "dot_product",
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 256},
            },
        }
    },
}


def main() -> None:
    es = Elasticsearch(ES_URL, request_timeout=120)

    if not es.ping():
        print(f"ERROR: Cannot reach Elasticsearch at {ES_URL}")
        print("Run 'make up' first and wait for the health check to pass.")
        sys.exit(1)

    info = es.info()
    print(f"Connected to Elasticsearch {info['version']['number']}")

    if es.indices.exists(index=INDEX_NAME):
        if "--delete" in sys.argv:
            es.indices.delete(index=INDEX_NAME)
            print(f"Deleted existing index '{INDEX_NAME}'")
        else:
            print(f"Index '{INDEX_NAME}' already exists. Pass --delete to recreate.")
            return

    es.indices.create(
        index=INDEX_NAME,
        settings=MAPPING["settings"],
        mappings=MAPPING["mappings"],
    )
    print(f"Index '{INDEX_NAME}' created.")
    print(
        f"  embedding_vector_v2 : {DIMS['v2']}d  (VGG19)    — HNSW m=16 ef_construction=100"
    )
    print(
        f"  embedding_vector_v3 : {DIMS['v3']}d  (pHash)    — HNSW m=16 ef_construction=100"
    )
    print(
        f"  embedding_vector_3d : {DIMS['3d']}d   (PointNet2) — HNSW m=16 ef_construction=100"
    )


if __name__ == "__main__":
    main()
