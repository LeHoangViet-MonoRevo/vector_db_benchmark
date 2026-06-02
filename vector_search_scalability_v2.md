# Vector Search Scalability — Working Report

**Author:** Le Hoang Viet  
**Last updated:** 2026-06-02  
**Status:** In progress — awaiting benchmark results before final recommendation

---

## Goal

Make the vector similarity search pipeline scalable and fast as the product catalogue grows beyond its current 100 K-document threshold.

The system must return ranked similarity results in **less than 1 s end-to-end** (excluding ML inference) for any company, regardless of catalogue size, with no upper bound on total document count.

---

## Constraints

| Constraint | Detail |
|---|---|
| Existing indexing pipeline must be preserved | The Kafka consumer (`SearchUpdaterProductCreation`) drives all product indexing. Changes to the data format or index structure must be backward-compatible or require a coordinated migration. |
| Per-org data isolation required | Search queries are always scoped to a single `organization_id`. Cross-tenant data leakage is unacceptable. |
| Rocchio feedback must continue to work | The feedback system modifies query vectors at runtime and maintains a cluster history in ES. The feedback pipeline is ES-native (Painless scripts, nested documents) and cannot be trivially moved. |
| Multi-vector search (v2 / v3 / 3D) | Three embedding types with different dimensions (4096 / 1024 / 512) must all be searchable. They use different models (VGG19, pHash, PointNet2) and different index strategies. |
| Latency | <1 s per search request |

**Expected scale:** Unlimited — scale as you go.

---

## Current Bottlenecks

### 1. `script_score` forces brute-force exact search (most critical)

When a user searches, the system asks Elasticsearch: "For every single product image belonging to this company, calculate how similar it is to my query image." It does this by running a dot product on every document, one by one — like flipping through every page of a phone book instead of using the index at the back.

```json
{
  "query": {
    "script_score": {
      "query": {
        "bool": {
          "filter": [
            {"term": {"version": "v2"}},
            {"term": {"organization_id": "42"}}
          ]
        }
      },
      "script": {
        "source": "dotProduct(params.query_vector, 'embedding_vector_v2')",
        "params": {"query_vector": [0.001, "...", "4096 values"]}
      }
    }
  }
}
```

What Elasticsearch does with this:
1. Evaluates the `bool` filter → collects all matching docs (every product of company 42 at version v2).
2. Iterates through **every** matched document and executes the Painless `dotProduct` script against it.
3. Maintains a priority queue of size 100 (`number_retrieval_vector`).

### 2. High-dimensional vectors amplify the cost

| Version | Model | Vector dims | Ops per doc |
|---|---|---|---|
| v2 | VGG19 | 4096 | 4096 |
| v3 | pHash | 1024 | 1024 |
| 3d | PointNet2 | 512 | 512 |

VGG19's 4096-dim embedding is the standard feature layer from a classification network. These dimensions are highly redundant for retrieval tasks. Each dimension adds directly to the linear scan cost.

### 3. O(n) per query vector × multiple crops per product

The system takes each cropped sub-image from the query drawing and creates a separate search for it. If a drawing has 5 cropped regions, it runs 5 full searches — each one scanning all 100 K+ products individually.

`_extract_crop_features` loops over every crop and builds one query vector each:

```python
for image in crop_base64:
    list_query.append(self.extract_feature(crop))  # one 4096-dim vector per crop
```

`search_vectors_batch` then issues one sub-query per vector as a single `_msearch` request:

> N crops → N sub-queries → each scans all 100 K docs → **N × 100 K dot products**

A project with 5 crops triggers 5 × 100 K = 500 K dot products against a 4096-dim index per search.

### 4. Single shared index, shard imbalance

All companies share `raijin_search_indexer`. The routing key bounds each query to one shard, but:

- If one company grows significantly faster than others, their shard becomes oversized.
- Elasticsearch rebalances shard count at index level, not routing-key level, so the hot shard cannot be automatically split away.

A 100 K product company occupies one shard containing ~200 K dense vector documents, each of which carries 3 embedded vectors totalling (4096 + 1024 + 512) × 4 bytes ≈ 22 KB of vector data alone per document. This is ~4.4 GB of raw vector data for the v2 field only at 100 K products.

### 5. Sequential network round-trips per search

Beyond the main vector search, each call to `ranking_project_ref` also makes:

| Step | ES operation | Size |
|---|---|---|
| Crop image fetch | `client.search` on `encoded_data_physical_object_2d` | Up to `size=10000` |
| Rocchio exact match | `client.search` on `rocchio_history_physical_object_1` | 1 doc |
| Disliked cluster fetch | `client.mget` on `similarity_clusters` | N clusters |
| Disliked phys ID lookup | `client.search` on `raijin_search_indexer` | Up to 10 K docs |
| V3 crop image fetch | `client.search` on `encoded_data_physical_object` | 1 doc |

These are sequential (not concurrent). Each is a network round-trip with its own serialisation/deserialisation overhead.

---

## System Architecture

System architecture stays unchanged since the problem is not in the architecture.

---

## Boss Feedback (2026-06-02)

> Comparing "Brute Force Elasticsearch (exact knn)" with "Milvus" is not a valid basis for decision-making. We cannot make an informed decision from this comparison alone.
>
> **Why:**
> The comparison is not being made at the same level. In addition, there is no validation, benchmark, or empirical data to support the proposal. Without comparable evaluation criteria and actual measurement results, we cannot make any architectural or technical decisions.
>
> **Detail:**
> 1. Why are we not considering an ANN-based architecture? Exact KNN does not seem like a scalable or efficient approach for this use case.
> 2. I do not understand why changing the infrastructure is being proposed as the solution when Elasticsearch native search methods, including HNSW, have not been tested at all.
> 3. Do not compare infrastructure options before comparing architectural approaches.
> 4. Please include benchmark data with actual numbers for cost, accuracy (recall), latency, throughput, and scalability. Without quantitative measurements, we cannot properly evaluate the trade-offs or make an informed decision.

### What the feedback means

The boss's objection has three layers:

**Layer 1 — Skipped the architectural comparison.**  
The HNSW index already exists in our mapping (`"index": True` on all three `dense_vector` fields). ANN search is available right now with zero infrastructure changes. The original report mentioned it as "Phase 1" but never tested it. The required question to answer first is: *"Is ES HNSW fast enough?"* — before asking *"Should we replace ES?"*

**Layer 2 — Compared at different algorithm levels.**  
The original report compared `script_score` (ES exact/brute-force) vs Milvus (which can be exact or ANN). That is not a fair comparison. It mixes the architectural decision (exact vs ANN) with the infrastructure decision (ES vs Milvus).

Valid comparison structure:

| Comparison | What it proves |
|---|---|
| ES exact (`script_score`) vs ES ANN (`knn` HNSW) | Is ANN acceptable? How much recall do we lose? |
| ES ANN vs Milvus ANN | Is the infrastructure switch worth it, at the same algorithm level? |

**Layer 3 — No empirical data.**  
All numbers in the original report were estimates. Real measurements from the actual index are required.

---

## Revised Decision Framework

The correct order is:

```
1. Architectural decision: exact KNN vs ANN?
   └─ Test ES HNSW (already built on disk) against ES script_score
      └─ If recall@100 >= 0.95 AND latency < 1s at target scale → ES HNSW is sufficient. Done.
      └─ If ES HNSW insufficient at target scale → proceed to infra comparison

2. Infrastructure decision (only if step 1 shows ES is insufficient):
   └─ ES HNSW (ANN) vs Milvus HNSW (ANN) — same algorithm level, different infra
      └─ Compare: latency, recall, throughput, cost, operational complexity
```

---

## Part 1 — Architectural Comparison: Exact KNN vs ANN

> **Status:** To be filled after benchmark runs.

---

## Part 2 — Benchmark Results

> **Status:** To be filled after benchmark runs.

### 2.1 Methodology

### 2.2 Test Environment

### 2.3 Latency

### 2.4 Recall

### 2.5 Throughput

### 2.6 Scalability

---

## Part 3 — Infrastructure Comparison: ES HNSW vs Milvus HNSW

> **Status:** To be filled only if Part 1 shows ES HNSW is insufficient.

### 3.1 Cost

### 3.2 Latency

### 3.3 Recall

### 3.4 Throughput

### 3.5 Scalability

### 3.6 Operational Complexity

---

## Next Steps (Benchmark Phase)

### Step 1 — Implement ES HNSW switch (code change, ~30 min)

Switch `search_vector()` and `search_vectors_batch()` in `app/db/vector_db/elasticsearch/interaction.py` from `script_score` to the native `knn` query DSL.

The HNSW graph is already built on disk. This activates it.

```python
# Current (brute-force, O(n)):
"query": {
    "script_score": {
        "query": {"bool": {"filter": [...]}},
        "script": {"source": "dotProduct(params.query_vector, 'embedding_vector_v2')", ...}
    }
}

# New (HNSW ANN, O(log n)):
"knn": {
    "field": "embedding_vector_v2",
    "query_vector": [...],
    "k": 100,
    "num_candidates": 200,
    "filter": [
        {"term": {"version": "v2"}},
        {"term": {"organization_id": str(organization_id)}}
    ]
}
```

Note: `knn` is a top-level key in the request body, not nested inside `"query"`.  
`num_candidates` is the main recall/latency tuning knob — higher = better recall, slower.

### Step 2 — Write and run benchmark script (~2 hours)

The benchmark must measure all four dimensions the boss requested:

| Metric | How to measure |
|---|---|
| **Latency** (p50, p95, p99) | `time.perf_counter()`, 50+ query runs per configuration |
| **Recall@100** | Run both exact and ANN on same query vectors; `len(set(knn_ids) & set(exact_ids)) / 100` |
| **Throughput** (QPS) | Concurrent requests, measure wall-clock time |
| **Scale behavior** | Repeat at N = 10 K, 50 K, 100 K docs |

Benchmark matrix:

| Configuration | Algorithm | Infra | Expected |
|---|---|---|---|
| Current | Exact KNN (`script_score`) | ES | Baseline |
| Phase 1 | ANN (`knn`, `num_candidates=200`) | ES HNSW | Measure |
| Phase 1 | ANN (`knn`, `num_candidates=500`) | ES HNSW | Measure |

### Step 3 — Present results to boss

If ES HNSW meets `<1 s` at target scale with `recall@100 >= 0.95`: **recommend ES HNSW, no Milvus**.

If ES HNSW is insufficient at target scale: add Milvus HNSW to the benchmark matrix at the same recall level, present side-by-side with cost and operational complexity.

---

## Original Scaling Strategy (on hold pending benchmarks)

The strategy below was the original proposal. It is preserved here for reference but should not be presented to the boss until Step 2 benchmark data is available to support or refute it.

### Phase 1 — Enable ES HNSW (days, zero infrastructure change)

**What:** Replace `script_score + dotProduct` with `knn` query DSL in `search_vector()` and `search_vectors_batch()`.

**Why:** The HNSW index is already built on disk (`"index": True` in the mapping). Zero data migration, zero new services.

**Estimated gain:** 10–50× on vector search latency. At 100 K docs: 500–3 000 ms → 10–50 ms. *(To be validated by benchmark.)*

**Risk:** Approximate results; slight accuracy loss (quantify with recall@100 from benchmark).

**Estimated scale ceiling:** ~500 K docs/company.

---

### Phase 1.5 — Reduce vector dimensions (1 week)

**What:** Research a solution to reduce the V2 embedding dimension from 4096 → 1024 (e.g., PCA, MRL, fine-tuning with a smaller head).

**Why:** V2 at 4096-dim is the primary bottleneck — 4× more expensive than V3, 8× more than 3D.

| Metric | 4096-dim | 1024-dim | Gain |
|---|---|---|---|
| Dot products per scanned doc | 4096 | 1024 | 4× less compute |
| VRAM per 100 K products | 3.2 GB | 0.8 GB | 4× less VRAM |
| Query vector payload per `_msearch` sub-query | 16 KB | 4 KB | 4× less I/O |

**Estimated scale ceiling:** ~300 K docs/company.

---

### Phase 2 — Milvus: CPU FLAT (1 week)

**What:** Stand up Milvus in standalone mode on CPU. Migrate all vectors from the ES index. Cut over `search_vectors_batch()` to Milvus.

**Why:** Milvus FLAT is implemented in optimised C++ with AVX-512 SIMD, scans vectors in column-oriented memory (cache-friendly), and parallelises across all CPU cores.

| Method | 100 K docs (2 crops) | 500 K docs |
|---|---|---|
| ES `script_score`, 4096-dim (current) | 500–3 000 ms | >10 s |
| Milvus FLAT, 1024-dim (Phase 1.5 applied) | 20–80 ms | 80–300 ms |

*(All numbers are estimates. Benchmark data from Phase 1 will inform whether this phase is necessary.)*

---

### Phase 3 — Milvus GPU (when GPU is provisioned, ~1 day)

**What:** Switch the Milvus index type from CPU FLAT to `GPU_BRUTE_FORCE`. No data migration — only an index rebuild on the same collection.

**Why:** `GPU_BRUTE_FORCE` runs the same exact dot products on thousands of CUDA threads in parallel — same 100% recall as Phase 2, ~10–50× faster.

**Estimated gain:** <1 ms per query at any scale up to available VRAM. QPS: 5 000–20 000 at batch size 1.

---

### Phase 4 — Milvus distributed cluster (when HA or multi-region is required)

**What:** Shard the Milvus collection across multiple Query Nodes when the corpus exceeds one machine's memory.

**Why:** At 1 B+ docs, no single machine holds all vectors in RAM/VRAM. Standalone Milvus hits a hard memory ceiling — cluster mode is the only way past it.

**Expected gain:** No latency improvement over Phase 3. The gain is capacity — the system can grow without a ceiling by adding nodes.

---

## Open Questions

- [ ] What is the actual corpus size per org today, and what is the projected growth rate?
- [ ] What is the acceptable recall@100 threshold for this use case? (95%? 99%?)
- [ ] Is there an existing dev/staging ES cluster against which benchmarks can be run without affecting production?
- [ ] If ES HNSW is sufficient, does the Rocchio feedback pipeline (`search_vector_w_filters` using `l2norm` script_score in `similarity_clusters`) also need to be switched to ANN?
