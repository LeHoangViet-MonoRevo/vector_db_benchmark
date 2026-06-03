# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start Elasticsearch + Kibana
make up

# Install dependencies and create index
make setup

# Load data at specific scale
make load-10k      # or load-50k, load-100k, load (all scales)

# Run benchmarks
make benchmark           # exact + ANN at all scales (slow, 100k exact takes minutes)
make benchmark-quick     # ANN only at 10k — fastest for iteration
make benchmark-ann       # ANN only across all scales (no recall numbers)

# Inspect cluster
make status
make count

# Full reset (deletes all volumes)
make reset
```

The Python interpreter is hardcoded to `/home/vietlh/miniconda3/envs/virenv1/bin/python` in the Makefile.

Direct script invocation:
```bash
python scripts/benchmark.py --scale 10k --no-exact
python scripts/load_data.py --scale 50k
python scripts/create_index.py --delete   # recreates index from scratch
```

## Architecture

This is a benchmarking suite comparing Elasticsearch **exact KNN** (`script_score` brute-force) against **HNSW ANN** (`knn` query) for vector similarity search.

### Why it exists

The production system (`raijin_search_indexer`) currently uses `script_score`, which does a full dot product scan on every document per query. At 100k products × 5 crop regions × 4096 dimensions, this is the current bottleneck. The benchmark tests whether ES's native HNSW can meet the production SLA (<1s latency, recall@100 ≥ 0.95) without migrating to Milvus.

### Data model

Three embedding types in one index (`raijin_search_indexer`):
- `embedding_vector_v2`: VGG19 features, 4096 dimensions
- `embedding_vector_v3`: pHash features, 1024 dimensions
- `embedding_vector_3d`: PointNet2 features, 512 dimensions

All vectors are **unit-normalized** so `dot_product` similarity equals cosine similarity. Queries are always scoped to a single `organization_id` (per-org isolation is a hard constraint).

### Index design

1 shard, 0 replicas; refresh interval set to 30s during bulk load then restored to 1s. HNSW params: `m=16`, `ef_construction=100`. Each benchmark scale uses a dedicated org ID (`bench_org_10k`, etc.) to prevent cross-contamination.

### Benchmark metrics (benchmark.py)

- **Latency**: p50/p95/p99 over 50 random queries
- **Recall@100**: overlap between exact and ANN result sets on identical query vectors
- **Throughput (QPS)**: wall-clock QPS under 10 concurrent workers via `ThreadPoolExecutor`
- **Scalability**: all metrics repeated at 10k, 50k, 100k scales

Key tuning knob: `NUM_CANDIDATES_VARIANTS = [200, 500]` in `config.py` — higher values improve recall at the cost of latency.

### Decision framework

1. If ES HNSW achieves recall@100 ≥ 0.95 AND latency < 1s → use HNSW, stop
2. Otherwise → evaluate Milvus migration (significant migration cost due to Rocchio feedback pipeline using ES Painless scripts)

See `vector_search_scalability_v2.md` for the full problem analysis.

## Infrastructure

Elasticsearch 9.0.0, `xpack.security.enabled=false`, `discovery.type=single-node`, JVM heap 2GB, ES at `http://localhost:9200`, Kibana at `http://localhost:5601`.
