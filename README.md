![AI-Memory Hackathon by cognee](hackathon-banner.jpg)

# What Happens When You Give an Analytics Agent a cognee search and ingested data?

> **TL;DR** — We built two identical LangGraph agents and gave them the same procurement analytics question. One searches **raw document chunks**. The other searches data that **cognee already processed into a knowledge graph** — summaries, entities, and relationships extracted at ingestion time. The question isn't "which agent is smarter?" — it's **"does investing in data quality at ingestion time pay off at query time?"**

---

## The Core Idea

### The Problem With RAG on Raw Documents

Most RAG systems work like this:

```
Documents → chunk → embed → vector DB → retrieve chunks → LLM reads chunks → answer
```

The LLM receives **raw text chunks** and has to figure out the structure on its own — every single query. If the data is messy JSON, the LLM parses it. If fields are inconsistent, the LLM handles it. All that work happens **at query time**, repeated for every question.

### Cognee's Approach: Do the Hard Work Once

Cognee adds a **cognify** step between ingestion and querying:

```
Documents → cognee.add() → cognee.cognify() → knowledge graph → query time is easy
                                │
                                ├── Summaries: natural language paragraphs per document
                                ├── Entities: 8,816 extracted (vendors, products, SKUs)
                                └── Relationships: 56,874 edges connecting entities
```

The key insight: **cognee.cognify() does entity extraction, summarization, and relationship mapping once at ingestion**. At query time, the agent searches pre-structured data instead of raw chunks.

### What We're Testing

> *"If both agents have the same LLM, tools, and embedding model — does the data cognee produced at ingestion time actually help at query time?"*

```
Same 2,000 documents (1,000 invoices + 1,000 transactions, 20 vendors)
Same LLM (GPT-4o-mini, temperature=0)
Same tool framework (LangGraph ReAct agent)
Same embedding model (nomic-embed-text-v1.5, 768 dims, local GGUF)
Same vector DB (Qdrant, localhost:6333)
```

The only variable: **what cognee produced vs. what was there before cognee**.

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

## Results

### Result 1: Search Relevance — Cognee Wins 75% of Queries

We ran 8 natural-language queries against both collections using the same embedding model and measured Qdrant similarity scores. **Cognee summaries returned more relevant results 75% of the time.**

| Query | Raw Score | Cognee Score | Delta | Winner |
|---|---|---|---|---|
| "vendors with the biggest discounts" | 0.659 | **0.744** | **+13%** | Cognee |
| "Who supplies monitors?" | 0.509 | **0.699** | **+37%** | Cognee |
| "networking equipment" | 0.461 | **0.576** | **+25%** | Cognee |
| "storage devices" | 0.458 | **0.538** | **+17%** | Cognee |
| "expensive high-value purchases" | 0.517 | **0.574** | **+11%** | Cognee |
| "small orders with low quantities" | 0.586 | **0.620** | **+6%** | Cognee |
| "Which transactions had the most products?" | **0.675** | 0.638 | -5% | Raw |
| "Find invoices from early 2025" | 0.674 | 0.681 | ~0% | Tie |

**Why?** The embedding model understands natural language better than JSON syntax:

```
Query: "Who supplies monitors?"

RAW top result (score 0.51):
  {'transaction_id': 'TX-V5-M07-530786', 'vendor_id': 5, 'amount': 971.919, 'items': "[{'product': '...

COGNEE top result (score 0.75):
  Purchase TX-V19-M06-901617 from vendor #19: 2 items — HP E27 G4 Monitor and Lenovo ThinkPad Keyboard...
```

The raw collection embeds JSON syntax (`{'vendor_id': 5, 'items': "[{...`). The cognee collection embeds natural language (`"HP E27 G4 Monitor"`). Embedding models are trained on natural language — **cognee.cognify() converts data into a format that embeds better**.

> Run `python test_search_quality.py` to reproduce this test.

### Result 2: Agent End-to-End Performance

We tested two queries — a focused analytics question and a complex multi-part question:

**Query A** (focused): *"Which vendor gives the highest average discount rate? Show top 5."*

| Metric | Raw Agent | Cognee Agent |
|---|---|---|
| **Typical time** | 30–75s | 35–45s |
| **Typical tool calls** | 2–5 | 2–3 |
| **run_analysis errors** | 0–3 (varies) | 0–1 (stable) |

**Query B** (complex): *"Analyze procurement spending: (1) Top 3 vendors by total spend, (2) what products do they supply and at what discount, (3) which vendor gives best value — lowest price per unit after discounts? Cite invoices."*

| Metric | Raw Agent | Cognee Agent |
|---|---|---|
| **Time** | 51.6s | 53.0s |
| **Tool calls** | 2 | 3 |
| **run_analysis errors** | 0 | 1 |

**Observations:**
- On any single run, both agents can perform similarly — a smart LLM compensates for messy data.
- The raw agent's results are **more variable**: sometimes 2 calls / 30s, sometimes 5 calls / 75s, depending on whether the LLM writes correct pandas code for the mixed schema (`total` vs `amount`, nested `items` strings).
- The cognee agent is **more consistent**: typically 2–3 calls, because flat uniform fields (`total`, `discount`, `discount_pct`) are simpler for the LLM to work with.
- Speed differences are mostly LLM API latency (~15s per round-trip to OpenAI), not search or computation.

### Result 3: What cognee.cognify() Actually Produced

This is the core of the experiment. cognee's ingestion pipeline ran **once** on the raw documents and produced:

| What cognee created | Count | What it enables |
|---|---|---|
| **TextSummary_text** | 2,000 | Natural language summaries that embed 13–37% better |
| **Entity_name** | 8,816 | Extracted vendors, products, SKUs as searchable entities |
| **Relationships** | 56,874 | Edges connecting entities (vendor→product, invoice→item) |

The raw agent has **none of this**. It searches the same 2,000 documents but as JSON blobs.

---

## Why This Matters

### The Honest Take

Both agents use the same LLM, and a smart LLM can compensate for bad data — it expands "computer memory" into "RAM DDR4 DDR5" in its search query, it retries when parsing fails. So on any single query, the raw agent can match the cognee agent.

But **cognee's value is cumulative**:

### 1. Better Embeddings From Better Text

This is the strongest finding. `cognee.cognify()` converts structured data into natural language summaries. Embedding models (trained on natural language) produce **13–37% more relevant vectors** from these summaries than from raw JSON. This means:
- Fewer irrelevant results returned
- Less data for the LLM to filter through
- Better answers from the same number of search results

### 2. Uniform Schema = Fewer Agent Errors

Raw data has inconsistent fields (`total` vs `amount`, `items` as nested strings). Cognee summaries normalize everything into flat, consistent fields. The LLM writes simpler code that fails less.

### 3. Entity Knowledge the Raw Agent Can't Access

The cognee agent can search 8,816 extracted entities — vendor names, product names, SKUs. The raw agent has no equivalent. For questions like "What products does Vendor 3 sell?", the cognee agent can do a targeted entity search instead of scanning all 2,000 documents.

### 4. The Real-World Argument

In production, documents are messy — PDFs, emails, varied formats. The question isn't "can a smart LLM parse this?" — it's "should it have to, every single time?"

```
Without cognee:  Every query → LLM parses raw data → errors → retries → slow
With cognee:     cognee.cognify() runs once → clean data forever → queries are simple
```

**cognee.cognify() is a one-time investment that pays off on every subsequent query.**

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
