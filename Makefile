PYTHON := /home/vietlh/miniconda3/envs/virenv1/bin/python

.PHONY: up down reset setup load load-10k load-50k load-100k load-200k load-300k load-500k \
        benchmark benchmark-quick \
        up-milvus down-milvus reset-milvus milvus-setup milvus-load \
        milvus-load-10k milvus-load-50k milvus-load-100k milvus-load-200k milvus-load-300k milvus-load-500k \
        milvus-benchmark milvus-benchmark-quick milvus-benchmark-ann milvus-status \
        visualize visualize-compare

up:
	docker compose up -d
	@echo "Waiting for Elasticsearch to be healthy..."
	@until curl -sf http://localhost:9200/_cluster/health?wait_for_status=yellow > /dev/null 2>&1; do \
		echo "  still waiting..."; sleep 5; \
	done
	@echo "Elasticsearch is ready."

down:
	docker compose down

reset:
	docker compose down -v
	docker compose up -d

setup:
	pip install -q -r requirements.txt
	$(PYTHON) scripts/create_index.py

setup-fresh:
	pip install -q -r requirements.txt
	$(PYTHON) scripts/create_index.py --delete

load:
	$(PYTHON) scripts/load_data.py --scale all

load-10k:
	$(PYTHON) scripts/load_data.py --scale 10k

load-50k:
	$(PYTHON) scripts/load_data.py --scale 50k

load-100k:
	$(PYTHON) scripts/load_data.py --scale 100k

load-200k:
	$(PYTHON) scripts/load_data.py --scale 200k

load-300k:
	$(PYTHON) scripts/load_data.py --scale 300k

load-500k:
	$(PYTHON) scripts/load_data.py --scale 500k

load-1m:
	$(PYTHON) scripts/load_data.py --scale 1m

# Full benchmark: exact + ANN at all scales (slow — 100k exact takes minutes)
benchmark:
	$(PYTHON) scripts/benchmark.py --scale all

# Exact up to 500k, HNSW-only for 1M
benchmark-mixed:
	$(PYTHON) scripts/benchmark.py --scale all --no-exact-above 500000

# Quick benchmark: skip exact KNN, 10k only — good for iteration
benchmark-quick:
	$(PYTHON) scripts/benchmark.py --scale 10k --no-exact

# ANN only at all scales (no recall numbers, but fast)
benchmark-ann:
	$(PYTHON) scripts/benchmark.py --scale all --no-exact

status:
	@curl -sf http://localhost:9200/_cat/indices?v || echo "Elasticsearch not running"

count:
	@curl -sf "http://localhost:9200/raijin_search_indexer/_count?pretty" || echo "Index not found"

# ── Milvus ──────────────────────────────────────────────────────────────────

up-milvus:
	docker compose -f docker-compose.milvus.yml up -d
	@echo "Waiting for Milvus to be healthy..."
	@until curl -sf http://localhost:9091/healthz > /dev/null 2>&1; do \
		echo "  still waiting..."; sleep 5; \
	done
	@echo "Milvus is ready."

down-milvus:
	docker compose -f docker-compose.milvus.yml down

reset-milvus:
	docker compose -f docker-compose.milvus.yml down -v
	docker compose -f docker-compose.milvus.yml up -d

milvus-setup:
	pip install -q pymilvus
	$(PYTHON) scripts/milvus_setup.py

milvus-setup-fresh:
	$(PYTHON) scripts/milvus_setup.py --delete

milvus-load:
	$(PYTHON) scripts/milvus_load.py --scale all

milvus-load-10k:
	$(PYTHON) scripts/milvus_load.py --scale 10k

milvus-load-50k:
	$(PYTHON) scripts/milvus_load.py --scale 50k

milvus-load-100k:
	$(PYTHON) scripts/milvus_load.py --scale 100k

milvus-load-200k:
	$(PYTHON) scripts/milvus_load.py --scale 200k

milvus-load-300k:
	$(PYTHON) scripts/milvus_load.py --scale 300k

milvus-load-500k:
	$(PYTHON) scripts/milvus_load.py --scale 500k

milvus-load-1m:
	$(PYTHON) scripts/milvus_load.py --scale 1m

# Full benchmark: exact FLAT + HNSW at all scales
milvus-benchmark:
	$(PYTHON) scripts/milvus_benchmark.py --scale all

# Quick: ANN only at 10k, no recall numbers
milvus-benchmark-quick:
	$(PYTHON) scripts/milvus_benchmark.py --scale 10k --no-exact

# ANN only at all scales (no recall numbers, faster)
milvus-benchmark-ann:
	$(PYTHON) scripts/milvus_benchmark.py --scale all --no-exact

milvus-status:
	@curl -sf http://localhost:9091/healthz && echo " Milvus healthy" || echo "Milvus not running"

# ── Visualisation ────────────────────────────────────────────────────────────

# ES only — saves JSON then renders PNG
visualize:
	$(PYTHON) scripts/benchmark.py --scale all --json results/es_results.json
	$(PYTHON) scripts/visualize.py results/es_results.json --output benchmark_report.png

# ES vs Milvus side-by-side comparison PNG
visualize-compare:
	$(PYTHON) scripts/visualize.py results/es_results.json results/milvus_results.json \
		--labels "ES" "Milvus" --output benchmark_compare.png
