ES_URL = "http://localhost:9200"
INDEX_NAME = "raijin_search_indexer"

MILVUS_HOST = "localhost"
MILVUS_PORT = 19530


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
SCALES = [10_000, 50_000, 100_000]


# Organization IDs used per scale
def org_id(scale: int) -> str:
    return f"bench_org_{scale // 1000}k"


# Number of query vectors per latency/recall run
N_QUERIES = 50

# Top-K results to retrieve
TOP_K = 100

# HNSW num_candidates variants to test
NUM_CANDIDATES_VARIANTS = [200, 500]

# Bulk indexing batch size
BULK_BATCH = 500

# Throughput test concurrency
CONCURRENCY = 10
