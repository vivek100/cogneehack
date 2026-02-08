![AI-Memory Hackathon by cognee](hackathon-banner.jpg)

# Cognee vs Raw: Does a Knowledge Graph Make AI Agents Better?

> **TL;DR** — We built two identical LangGraph agents and gave them the same procurement analytics questions. One agent searches **raw JSON documents**. The other searches **cognee's pre-processed knowledge graph** (summaries, entities, relationships). Same LLM, same tools, same data — different data quality. The results show how structured knowledge changes agent behavior.

---

## The Experiment

### Question We're Answering

> *"If I run the same AI agent on raw documents vs. cognee-structured data, what's the difference in speed, accuracy, and reliability?"*

### Setup

```
Same 2,000 documents (1,000 invoices + 1,000 transactions, 20 vendors)
Same LLM (GPT-4o-mini, temperature=0)
Same tool framework (LangGraph ReAct agent)
Same embedding model (nomic-embed-text-v1.5, 768 dims, local GGUF)
Same vector DB (Qdrant, localhost:6333)
```

The only difference: **what data each agent searches**.

---

## Architecture

### How It Works

```
                    ┌─────────────────────────────────────────────┐
                    │              User Question                  │
                    │  "Which vendor gives the highest discount?" │
                    └──────────────────┬──────────────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
              ┌──────────────────┐        ┌──────────────────┐
              │    RAW AGENT     │        │  COGNEE AGENT    │
              │  (baseline)      │        │  (knowledge graph)│
              └────────┬─────────┘        └────────┬─────────┘
                       │                           │
                       ▼                           ▼
              ┌──────────────────┐        ┌──────────────────┐
              │   raw_search     │        │ cognee_search_   │
              │                  │        │ summaries        │
              │ Searches:        │        │                  │
              │ DocumentChunk_   │        │ Searches:        │
              │ text (2,000)     │        │ TextSummary_text │
              │                  │        │ (2,000) +        │
              │ Raw JSON blobs   │        │ Entity_name      │
              │ with nested      │        │ (8,816)          │
              │ items-as-string  │        │                  │
              └────────┬─────────┘        │ Pre-parsed       │
                       │                  │ structured dicts │
                       │                  └────────┬─────────┘
                       │                           │
                       ▼                           ▼
              ┌──────────────────────────────────────────────┐
              │              search_results dict              │
              │  (pre-parsed, injected into sandbox)         │
              └──────────────────┬───────────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────────────┐
              │              run_analysis                     │
              │  Python sandbox (pandas, numpy, json)        │
              │  Accesses search_results directly            │
              │  No data copy-paste through LLM context      │
              └──────────────────┬───────────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────────────┐
              │              Computed Answer                  │
              │  With citations: vendor IDs, amounts, dates  │
              └──────────────────────────────────────────────┘
```

### The Key Innovation: Data Store Pattern

Most agent frameworks pass tool results back through the LLM context. This means:
1. Search returns 50 records → LLM reads all of them (thousands of tokens)
2. LLM writes Python code that **hardcodes the same data** → tokens wasted twice
3. If parsing fails, LLM retries → even more tokens

**Our approach**: Search tools store results in a shared `data_store`. The `run_analysis` sandbox gets them pre-injected as `search_results`. The LLM only sees a short preview + field names.

```
┌─────────────┐     preview (3 records)      ┌─────────┐
│ Search Tool ├─────────────────────────────►│   LLM   │
│             │                              │         │
│  stores     │     full data (50 records)   │ writes  │
│  results ───┼─────────────────────────────►│ simple  │
│  in dict    │     search_results dict      │ pandas  │
└─────────────┘     (injected into sandbox)  │ code    │
                                             └────┬────┘
                                                  │
                                                  ▼
                                        ┌─────────────────┐
                                        │  run_analysis    │
                                        │  sandbox         │
                                        │                  │
                                        │  search_results  │
                                        │  already loaded  │
                                        └─────────────────┘
```

---

## What Each Agent Sees

### Raw Agent: Messy JSON Blobs

```json
{
  "transaction_id": "TX-V5-M07-530786",
  "date": "2025-07-14",
  "vendor_id": 5,
  "amount": 971.92,
  "items": "[{'product': 'Seagate 4TB External HDD', 'sku': 'MB-HDD-005', 'qty': 9, 'price': 119.99, 'total': 1079.91}]",
  "discount": 107.99
}
```

Problems:
- `items` is a **string containing a Python dict** (not JSON) — needs `ast.literal_eval`
- Single quotes instead of double quotes — `json.loads()` fails
- Nested structure: items inside items
- No product categorization — "Seagate 4TB External HDD" is just a string

### Cognee Agent: Pre-Structured Summaries

```json
{
  "doc_id": "TX-V5-M07-530786",
  "doc_type": "transaction",
  "vendor_id": 5,
  "date": "2025-07-14",
  "total": 5585.32,
  "discount": 485.68,
  "discount_pct": 20.0,
  "net": 5585.32,
  "products": [{"qty": 9, "name": "Seagate 4TB External HDD"}],
  "summary": "Purchase TX-V5-M07-530786 (2025-07-14) from vendor 5..."
}
```

Advantages:
- **Already parsed** — vendor_id, total, discount are direct fields
- **Discount percentage** pre-calculated
- **Products** extracted as clean list
- **Human-readable summary** for context
- cognee did the hard work during ingestion — the agent just queries

---

## Test Results

### Test Query: *"Which vendor gives the highest average discount rate? Compare the top 5 vendors."*

This is a complex analytical question requiring: searching for discount data across vendors, computing per-record discount rates, aggregating by vendor, and ranking — exactly the kind of question where data quality matters.

#### Final Results (with Data Store + Pre-Parsing)

| Metric | Raw Agent | Cognee Agent | Winner |
|---|---|---|---|
| **Total time** | 74s | **46s** | Cognee (38% faster) |
| **Tool calls** | 5 | **3** | Cognee (40% fewer) |
| **Search time** | ~188ms | ~224ms | Comparable |
| **run_analysis retries** | 3 | 1 | Cognee |
| **Workflow** | search → 3 retries → answer | search → 1 retry → answer | Cognee |

**Why Cognee wins**: The raw agent's pre-parsed data still has mixed schemas (invoices have `total`, transactions have `amount`) and nested `items` lists — the LLM struggles to write correct pandas code on the first try. The cognee agent gets flat, uniform fields (`total`, `discount`, `discount_pct`) that are trivial to aggregate.

#### Evolution: How We Got Here

| Version | Raw Agent | Cognee Agent | Problem |
|---|---|---|---|
| **v1: No data store** | 88s / 7 calls | 109s / 5 calls | LLM copy-pastes data into code, fails at parsing |
| **v2: Data store, no pre-parse** | 30s / 2 calls | 31s / 2 calls | Works but LLM still writes parsing code |
| **v3: Data store + pre-parse** | 74s / 5 calls | **46s / 3 calls** | Complex query exposes schema differences |

The progression shows that **data quality compounds** — on simple queries both agents are equal, but on complex analytics the cognee agent's cleaner data leads to fewer errors and faster completion.

```
RAW AGENT workflow (5 tool calls):
  Step 1: raw_search("discount", limit=200)           →  188ms
  Step 2: run_analysis(pandas code)                    →  ERROR (schema mismatch)
  Step 3: run_analysis(fixed code)                     →  ERROR (next() usage)
  Step 4: run_analysis(fixed code)                     →  ERROR (lambda issue)
  Step 5: run_analysis(final fix)                      →  SUCCESS (210ms)

COGNEE AGENT workflow (3 tool calls):
  Step 1: cognee_search_summaries("all invoices", 1000) →  224ms
  Step 2: run_analysis(pandas code)                     →  SUCCESS (but wanted more detail)
  Step 3: run_analysis(refined code)                    →  SUCCESS (48ms)
```

---

## Why Cognee Matters

The cognee agent is **38% faster** and uses **40% fewer tool calls** on complex analytics. The advantage comes from **data quality at ingestion time**:

### 1. Better Search Relevance

Searching "discount" in summaries returns records that **explicitly mention discounts** in natural language. Searching "discount" in raw JSON matches on the `discount` field name — less semantically meaningful.

### 2. Richer Data Per Record

| Field | Raw Agent | Cognee Agent |
|---|---|---|
| Vendor ID | `vendor_id: 5` | `vendor_id: 5` |
| Amount | `amount: 971.92` | `total: 5585.32` |
| Discount | `discount: 107.99` | `discount: 485.68` + `discount_pct: 20.0` |
| Products | Nested string needing double-parse | Clean list of `{qty, name}` |
| Context | None | Full summary paragraph |

### 3. Entity Knowledge (8,816 entities)

The cognee agent has access to `Entity_name` — 8,816 extracted entities like vendor names, product names, SKUs, invoice numbers. This enables questions like:
- "What products does Vendor 3 sell?" → entity search
- "Find all Dell laptops" → entity search by product name
- "Which SKUs are associated with monitors?" → entity search by type

### 4. The Real-World Advantage

In production, raw documents are messy — PDFs, emails, varied formats. Cognee's ingestion pipeline (`cognee.add()` + `cognee.cognify()`) normalizes everything into a knowledge graph. The agent doesn't need to handle format variations — cognee already did.

```
Raw world:  PDF → OCR → messy text → agent struggles to parse
Cognee:     PDF → cognee.add() → cognee.cognify() → clean entities + summaries → agent just queries
```

---

## Project Structure

```
ai-memory-hackathon/
├── agents/
│   ├── cognee_orchestrator.py    # Cognee agent — searches summaries + entities
│   └── raw_orchestrator.py       # Raw agent — searches DocumentChunk_text only
├── tools/
│   ├── cognee_tools.py           # cognee_search_summaries, cognee_search_entities
│   ├── raw_tools.py              # raw_search (with pre-parsing)
│   └── shared_tools.py           # run_analysis (Python sandbox), explore_data
├── shared/
│   ├── qdrant_setup.py           # Qdrant client, local embedding model, cognee patches
│   └── data_store.py             # Shared dict for search→run_analysis data flow
├── run_agents.py                 # CLI runner for both agents
├── test_agents.py                # Side-by-side comparison test
└── langgraph.json                # LangGraph agent registration
```

---

## How to Run

```bash
# Prerequisites: Docker running with Qdrant, .env with OPENAI_API_KEY

# Install dependencies
uv sync
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Run the comparison test
uv run python test_agents.py

# Run interactively
uv run python run_agents.py --both

# Run just one agent
uv run python run_agents.py --raw
uv run python run_agents.py --cognee

# Run via LangGraph dev server
uv run langgraph dev
```

---

## Technical Details

### Embedding Dimension Fix

Cognee 0.4.1 has a bug: its `LiteLLMEmbeddingEngine.embed_text()` calls `litellm.aembedding()` without passing `dimensions=768`, so OpenAI returns 1536-dim vectors but Qdrant expects 768-dim. We monkey-patched this in `shared/qdrant_setup.py`.

### Kuzu Read-Only Mode

Cognee uses Kuzu as its graph database. Since multiple processes may access it, we monkey-patched Kuzu to open in read-only mode to avoid lock conflicts.

### Data Store Pattern

Search tools call `data_store.save(key, items)`. The `run_analysis` sandbox gets `data_store.get_all()` injected as `search_results` in its globals. The LLM never sees the full data — just a preview with field names and a sample record.

---

## Original Hackathon Instructions

*The sections below are from the original hackathon README.*

---

## What You Work With

- Query Cognee using natural-language questions (see how `completion` is generated in the `solution_q_and_a.py`).
- Receive structured or free-text answers.
- Use those answers, however, you like in your project.

## Constraints

- Qdrant must be the vector store of choice – whether local or hosted.
- The local model must remain functional; online LLM use is optional.
- The raw data included in the `data` folder is there for reference and should not be used directly.

## What You Can Build

Any tool, workflow, interface, or feature that benefits from QA over vendor, product, payment, or order information.

## Deliverables

- Create a folder named `submission` on your USB stick and place your entire project inside it. Alternatively, you can share your GH repo with [luca@topoteretes.com](mailto:luca@topoteretes.com)
- Your project must include code that demonstrates successful queries to Cognee.
- Be ready to give a short demo.

## Notes

- You do not have to add new files, modify or enrich the graph. In case you want to, there is some additional data in the `optional_data_for_enrichment` folder.

## Setup for Q&A with Qdrant and Local Model

**We will set up**:
- Ollama with two local models (embedding and LLM)
- A Python virtual environment with pinned dependencies
- A Cognee knowledge graph imported from prebuilt data
- A local Qdrant vector store loaded with snapshot data
- The question answering script (`solution_q_and_a.py`)

**This will allow you to**:
- Access ingested data from invoice and transaction documents
- Retrieve structured context from a knowledge graph for LLM queries
- Ask natural-language questions about the data using a local language model
- Build tools, agents, or workflows on top of the Q&A pipeline

**Before installation**:
- copy `models/` from the USB to your working directory
- do the same for `cognee_export/`
- verify the three subdirectories contain Modelfile and a *.gguf each

**Project installation**:
```bash
# Ollama installation
brew install ollama   # Mac OS
ollama serve &

# Ollama model registration
cd models
ollama create nomic-embed-text -f nomic-embed-text/Modelfile
ollama create cognee-distillabs-model-gguf-quantized -f cognee-distillabs-model-gguf-quantized/Modelfile
cd ..

# Initialize python environment, install dependencies
uv venv
source .venv/bin/activate
uv sync

# Graph setup
python setup.py

# Qdrant (local Docker)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Configure for use locally, retrieve data, restore to database
cp .env.example.local .env
uv run python download-from-spaces.py
uv run python restore-snapshots.py

# Run Q and A example
python solution_q_and_a.py
```

**Pitfalls to avoid**:
- failing to copy both `models/` and `cognee_export/` from USB
- building the venv in `models/` instead of the project root
- having a stale venv activated
- Ollama is not running
- New Qdrant conflicting with old in Docker

**Next steps**:
- look around the code
- play with the queries
- check out the databases
- build something

## Useful setup commands

Skip this reference if setup went smoothly.

**Turn off and remove Qdrant from Docker**

If necessary for recreating:
```bash
docker stop qdrant && docker rm qdrant
docker volume rm qdrant_storage
```

**Mac start/stop ollama**
```bash
brew services start ollama
brew services stop ollama
brew services info ollama
```

**Linux start/stop ollama**
```bash
sudo systemctl start ollama
sudo systemctl stop ollama
sudo systemctl status ollama
```

**Alternate Ollama Installation**
```bash
# Alternate direct option
curl -fsSL https://ollama.com/install.sh | sh
```

## What data do I have?

**After restore**, your cluster contains 14,837 vectors across 6 collections:

| Collection | Records | Content |
|---|---|---|
| DocumentChunk_text | 2,000 | Invoice and transaction chunks |
| Entity_name | 8,816 | Products, vendors, SKUs |
| EntityType_name | 8 | Entity type definitions |
| EdgeType_relationship_name | 13 | Relationship types |
| TextDocument_name | 2,000 | Document references |
| TextSummary_text | 2,000 | Document summaries |

These items are also connected via semantics in your graph DB.

The models included in the `models/` directory:
- **nomic-embed-text** -- 768-dim embeddings, local inference
- **Distil Labs SLM** -- fine-tuned reasoning model, GGUF quantized
- **Qwen3-4B** -- fallback LLM, optional

## Example Project Architecture

Several example projects which one can work off of (if desired, totally optional). These are three ready-to-run FastAPI projects: semantic search, spend analytics, and anomaly detection on procurement data.

**Stack:** [cognee](https://github.com/topoteretes/cognee) (knowledge graph memory) + [Qdrant Cloud](https://cloud.qdrant.io) (vector search) + [Distil Labs](https://www.distillabs.ai/) (LLM reasoning) + [DigitalOcean](https://www.digitalocean.com/) (deployment)

```
Raw documents
    |
    v
cognee.add() + cognee.cognify()     <-- cognee extracts entities, relationships, summaries
    |
    v
Qdrant Cloud (6 collections)        <-- vectors + knowledge graph stored here
    |
    v
FastAPI apps                         <-- search, analytics, anomaly detection
    |
    v
Distil Labs SLM                     <-- LLM reasoning (local GGUF or hosted API)
    |
    v
DigitalOcean App Platform           <-- deployed and shareable
```

## Example projects

Hackathon participants should feel free to build off of these if they wish, or to do something totally different. The three example projects in project1, project2, and project3 directories are each self-contained with their own`pyproject.toml` and dependencies:
```bash
cd project1-procurement-search  # or project2 or project3
uv sync
uv run python app.py
```

**Project 1: Procurement Semantic Search** (port 7777) -- semantic search across all procurement data with interactive UI.

**Qdrant features:** Query API, Prefetch + RRF Fusion, Group API, Discovery API, Recommend API, payload indexing, filtered search

**Endpoints:** `/search`, `/search/grouped`, `/discover`, `/recommend`, `/filter`, `/ask` (RAG Q&A), `/cognee-search`, `/add-knowledge`, `/collections`

**Project 2: Spend Analytics Dashboard** (port 5553) -- interactive analytics dashboard with Chart.js visualizations and semantic search.

**Qdrant features:** Scroll API (bulk extraction), Query API, Group API, payload indexing

**Endpoints:** `/api/analytics`, `/api/search`, `/api/search/grouped`, `/api/insights` (LLM analysis)

**Project 3: Anomaly Detective** (port 6971) -- automated anomaly detection using vector analysis and Qdrant's Batch Query API. Detection methods include amount outliers (z-score), embedding outliers (centroid distance), near-duplicates (similarity > 0.99), and vendor variance.

**Qdrant features:** Batch Query API (50 recommend queries/request), Recommend API, Scroll API with vectors, payload indexing

**Endpoints:** `/api/anomalies`, `/api/search`, `/api/investigate/{point_id}`, `/api/explain/{point_id}` (LLM explanation)

## Using Qdrant Cloud (alternative to local Docker)

If you prefer hosted Qdrant over local Docker, set up a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io) and use `.env.example` instead of `.env.example.local`:
```bash
cp .env.example .env
# Edit .env -- fill in QDRANT_URL and QDRANT_API_KEY with your Cloud values
uv run python download-from-spaces.py
uv run python restore-snapshots.py
```

## Example results

Example results comparing LLM and SLM outputs can be found in `responses.txt`.

## Adding your own data

The starter data was built using cognee's ECL (Extract, Cognify, Load) pipeline:
```bash
cd cognee-pipeline
cp .env.example .env
# Edit .env: add Qdrant credentials + LLM provider
uv sync
uv run python ingest.py
```

Programmatic usage:
```python
import cognee
from cognee.api.v1.search import SearchType

await cognee.add("Your document text here...")
await cognee.cognify()
results = await cognee.search(
    query_text="What vendors supply IT equipment?",
    query_type=SearchType.CHUNKS,
)
```

Supported input types: plain text strings, PDF, DOCX, TXT, CSV files, URLs, and directories of files.

To reset and re-ingest from scratch:
```python
await cognee.prune.prune_data()
await cognee.prune.prune_system(metadata=True)
```

See [cognee docs](https://docs.cognee.ai) for full pipeline options.


## Using Qwen3 as an alternative model

Register the Qwen3 model with Ollama:
```bash
cd models
ollama create Qwen3-4B-Q4_K_M -f Qwen3-4B-Q4_K_M/Modelfile
cd ..
```

Access it via the standard OpenAI-compatible interface at `http://localhost:11434/v1` with model name `Qwen3-4B-Q4_K_M`.


## DigitalOcean deployment

Two modes are available: **local** (GGUF models, default) and **remote** (API-based inference).

**Local dev** runs the Distil Labs SLM via llama-cpp-python (requires 4-8GB RAM):
```bash
# .env: LLM_MODE=local, EMBED_MODE=local (defaults)
uv run python app.py
```

**Remote deployment** to DigitalOcean App Platform:
```bash
uv run python upload-to-spaces.py
# Set LLM_MODE=remote and EMBED_MODE=remote in .env
doctl apps create --spec .do/app.yaml
```

Or run remotely via Docker:
```bash
docker compose up
```

**Environment variables**:

| Variable          | Default           | Description                                 |
| ----------------- | ----------------- | ------------------------------------------- |
| `QDRANT_URL`      | -                 | Qdrant Cloud cluster URL                    |
| `QDRANT_API_KEY`  | -                 | Qdrant Cloud API key                        |
| `LLM_MODE`        | `local`           | `local` (GGUF) or `remote` (API)            |
| `LLM_API_URL`     | -                 | OpenAI-compatible chat completions endpoint |
| `LLM_API_KEY`     | -                 | API key for remote LLM                      |
| `LLM_MODEL_NAME`  | `distil-labs-slm` | Model name for remote LLM                   |
| `EMBED_MODE`      | `local`           | `local` (GGUF) or `remote` (API)            |
| `EMBED_API_URL`   | -                 | OpenAI-compatible embeddings endpoint       |
| `EMBED_API_KEY`   | -                 | API key for remote embeddings               |
| `SPACES_ENDPOINT` | -                 | DO Spaces endpoint                          |
| `SPACES_BUCKET`   | -                 | DO Spaces bucket name                       |


## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- Docker (for local Qdrant)


## Useful commands
