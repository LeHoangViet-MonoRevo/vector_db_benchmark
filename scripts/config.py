ES_URL = "http://localhost:9200"
INDEX_NAME = "raijin_search_indexer"

MILVUS_HOST = "localhost"
MILVUS_PORT = 19530


def es_index_name(scale: int) -> str:
    """Per-org index name — one index per scale, no cross-org HNSW contamination."""
    return f"raijin_bench_{scale // 1000}k"


def milvus_collection_name(scale: int, flat: bool = False) -> str:
    suffix = "_flat" if flat else ""
    return f"raijin_milvus_{scale // 1000}k{suffix}"


# Vector dimensions matching production models
DIMS = {
    "v2": 4096,  # VGG19
    "v3": 1024,  # pHash
    "3d": 512,  # PointNet2
}

# Benchmark scales — docs per org
SCALES = [10_000, 50_000, 100_000, 200_000, 300_000, 500_000, 1_000_000]


# Organization IDs used per scale
def org_id(scale: int) -> str:
    return f"bench_org_{scale // 1000}k"


# Number of query vectors per latency/recall run
N_QUERIES = 50

# Top-K results to retrieve
TOP_K = 100

# HNSW num_candidates / ef variants to test
# ES:    num_candidates — tested against pre-filtered org subset
# Milvus: ef           — tested against the full standalone collection
NUM_CANDIDATES_VARIANTS = [200, 500]   # ES (pre-filtered, needs nc ≥ k/selectivity)
MILVUS_EF_VARIANTS      = [200, 500]   # Milvus (standalone collection, ef > k=100 is sufficient)

# Bulk indexing batch size
BULK_BATCH = 500

# Throughput test concurrency
CONCURRENCY = 10
