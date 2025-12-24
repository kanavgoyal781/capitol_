## Project Overview

Hi 👋  

This project implements a production-grade data ingestion and semantic search pipeline designed to transform raw, unstructured webhook payloads into a structured, queryable knowledge base. Built on FastAPI, Qdrant, and Docker and deployed on Render, the system automates the full lifecycle of data processing. It enforces strict data contracts using Pydantic for schema validation, cleans HTML content, generates vector embeddings via OpenAI, and indexes them for high-speed semantic retrieval.

Unlike standard ingestion scripts, this pipeline is engineered for fault tolerance and data reliability in distributed environments. It utilizes a defensive "Bouncer Pattern" to validate payloads individually, ensuring that a single malformed record does not crash the entire processing batch. Critical architecture decisions include an idempotent upsert mechanism to handle duplicate webhooks gracefully and a Dead Letter Queue (DLQ) that quarantines failed records for debugging, ensuring "zero data loss" even when upstream data is corrupted.

To maintain operational visibility, the system is instrumented with structured logging and automated telemetry. After every ingestion run, the pipeline generates a health dashboard and CSV audit trail that tracks success rates, failure breakdowns, and root causes (e.g., missing URLs vs. Pydantic validation errors). This approach treats observability as a first-class citizen, allowing engineers to monitor pipeline health in real-time and triage data quality issues immediately.

If anything is unclear, breaks, or you’d like to discuss design choices, please feel free to reach out:

- **Email:** kanavgoyal@uchicago.edu  
- **Phone:** +1-312-287-2109

---

## 3 ways to run the pipeline
1. **Live Deployment** expalained in below in **1. Live deployment – how to run the hosted app** This requires giving the json as an input to the REST APIs.
2. **Running the FastAPI locally** explained below in **3.1 Start the API Locally** -  This requires giving the json as an input to the REST APIs.
3. **Running the pipeline through local python code** explained in **3.2 Run the Batch Transformation Pipeline (Writes to output/)** - This reads the input file and produces the output file locally.


## 1. Live deployment – how to run the hosted app

A live deployment of this API is available at:

- **Swagger docs:** https://capitol.onrender.com/docs  

From this UI you can exercise the full pipeline:

1. **Transform** raw customer API documents → Qdrant format  
   - `POST /pipeline/transform`
2. **Embed** the transformed documents (adds a `vector` field)  
   - `POST /pipeline/embed`
3. **Index** into the vector database (Qdrant)  
   - `POST /pipeline/index`
4. **Search** the vector database  
   - `GET /search?query=...&k=3`
5. **Run full pipeline in one call**  
   - `POST /pipeline/run_full`

> **Note:** The first request may take a couple of minutes while the Render container and Qdrant warm up.

---

## 2. Repository layout

From the repo root:

```text
.
├── app.py              # FastAPI app (transform → embed → index → search)
├── pipeline.py         # Batch transformer with logging + dead-letter queue
├── embedding_v3.py     # Embedding client (OpenAI)
├── vectordb_v3.py      # Qdrant vector DB wrapper
├── Dockerfile          # Container image for deployment (used on Render)
├── requirements.txt
├── data/
│   ├── raw_customer_api.json
│   ├── raw_sample.json
│   └── qdrant_schema.md
├── output/             # Logs, transformed data, dead-letter queue
└── tests/
    ├── test_contract.py      # Schema / contract tests
    ├── test_integration.py   # End-to-end flow tests
    └── test_properties.py    # Property-based / edge-case tests
```
---

## 3. How to Run Locally 🛠️

### Python version - python-3.13.5

Running the app locally provides better observability, allowing you to monitor real-time logs from terminal. On the live deployment, these internal logs are not publicly visible (though they are accessible via the Render dashboard). 

> **Note:** `app.py` prints real-time logs in terminal but does **not** write any files to disk.  
> To save the clean JSON output, dead-letter queue, and `ingestion_report.csv`, you need to run `pipeline.py` (described below).

### 3.1 Start the API Locally
```bash
git clone [https://github.com/YOUR-USERNAME/capitol-ai-assessment.git] #Clone the repository
cd capitol-ai-assessment  # or the folder where app.py lives

python -m venv .venv #create and activate a virtual env
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip 
pip install -r requirements.txt  #Install dependencies

export OPENAI_API_KEY="sk-..."  #Set your OpenAI API key (required for embeddings)

docker run --rm -p 6333:6333 qdrant/qdrant #Start a local Qdrant instance (vector DB)

python app.py  #Run the app
```

> **Note:** Open - http://localhost:8000/docs – to try the same endpoints as the deployed version.

### 3.2 Run the Batch Transformation Pipeline (Writes to output/)

Running the pipeline below will save the output into the capitol-ai-assessment/output folder:
	1.	Clean JSON output (processed output file)
	2.	ingestion_report.csv
	3.	Dead-letter queue

To run the data transformation pipeline, make sure the capitol-ai-assessment/data folder contains a file named raw_customer_api.json, then run:

```bash 
python pipeline.py # Run from repo root
```
### 3.3 Testing

I have added testing funtionality in this app. In order to run the tests, run:-
``` bash
# From the repo root
pytest -q          # run all tests (quiet)

# Optional: see detailed output / prints
pytest -sv         # verbose + show print/log output
```
This runs:

1. **Contract tests** – `tests/test_contract.py`  
2. **Integration tests** – `tests/test_integration.py`  
3. **Property tests** – `tests/test_properties.py`  

> Note: The exact behavior of each test suite is described later in the individual test sections.
## 4. Project Overview 📖

This solution implements a production-grade data ingestion pipeline that takes messy, nested CMS API data and turns it into clean, schema-validated Qdrant documents. It then embeds and indexes them into a Qdrant vector database for semantic search.

I primarily focused on **Path A: Data Engineering Depth**, transforming the full dataset (50 documents) with robust validation and error handling in line with `data/qdrant_schema.md`.

In addition, I implemented several elements from **Path B: Infrastructure & Deployment**:
* A deployable FastAPI service with clear endpoints.
* Dockerized app, deployed live on Render.
* Integration with a real Qdrant vector DB and OpenAI embeddings.
* Structured logging and a simple health/root endpoint.

### 🔧 Core Data Flow

At a high level, the system performs three main actions:
1.  **Transform** raw CMS API documents → Qdrant-ready `{ text, metadata }`.
2.  **Embed** the text field using OpenAI into a dense vector.
3.  **Index & Search** via Qdrant for semantic retrieval.

#### 1. Text Extraction
* Iterates through the deeply nested `content_elements` array.
* Filters for `text`-based elements.
* Uses `BeautifulSoup` to strip HTML and normalize content.
* Concatenates segments into a single, readable `text` field that is used for embeddings.

#### 2. Metadata Extraction
* Maps complex API fields into the Qdrant metadata schema: `title`, `url`, `external_id`, `publish_date`, `first_publish_date`, `datetime`, `website`, `sections`, `categories`, `tags`, `thumb`.
* Flattens nested taxonomy structures into simple lists.
* Extracts thumbnails from `promo_items`, converting relative `resizeUrl` paths into absolute URLs when `canonical_website` is available.

#### 3. Edge Case Handling
* Handles missing/nullable fields (`canonical_website`, `taxonomy`, dates, etc.) gracefully.
* Optional fields fall back to `None` or empty lists instead of breaking the run.
* **Mandatory failures** (no `_id`, no resolvable `url`, or empty `text`) cause the document to be skipped and sent to the **Dead Letter Queue** rather than crashing the pipeline.
* If a new document has an external ID that has been seen before, then the previous record gets overwritten with the new record

### 🛡️ Key Technical Features

#### 1. Robustness & Defensive API Handling
Real-world APIs often contain "garbage" entries (strings, nulls, mixed types inside arrays). At the FastAPI layer (`app.py`):
* Each endpoint accepts `List[Any]` and validates each item with `isinstance(doc, dict)`.
* Non-dict items are skipped, logged, and written to the Dead Letter Queue.
* **Result:** One bad item never takes down the entire batch or API.

#### 2. Data Quality Validation (Pydantic)
All successfully transformed documents are validated using Pydantic models:
* `MetadataModel` enforces the metadata schema.
* `QdrantDocument` ensures the final `{ text, metadata }` structure matches expectations.
* If a document passes this step, it is guaranteed to conform to the Qdrant schema (types, required fields, and structure).

#### 3. Dead Letter Queue – No Silent Failures
Any document that fails a critical step (ID/URL/text missing, schema validation error, malformed structure) is written to `output/dead_letter_queue.jsonl`. Each entry includes:
* The document ID (if available).
* A human-readable reason.
* The raw document payload.

This makes failures auditable and debuggable instead of silent.

#### 4. Granular Observability
The pipeline is designed to be transparent.
* **Console + File Logs** (`output/pipeline.log`) show per-document traces (e.g., whether text, title, dates, or tags were found).
* **Ingestion Summary:** `output/ingestion_report.csv` provides a CSV report with `id`, `status`, and `reason`.
* **Processed Output:** Successfully transformed documents are saved as JSON (e.g., `output/processed_output.json`) for offline inspection.

#### 5. Embeddings, Vector DB & Live Deployment
* Embeddings are generated via OpenAI in `embedding_v3.py`.
* Documents are indexed into Qdrant via a small wrapper in `vectordb_v3.py`.
* The full flow (transform → embed → index → search) is exposed as a FastAPI service, containerized with Docker, and deployed on Render with a public `/docs` UI.

#### 6. Testing
A small but meaningful test suite ensures the system behaves as expected:
* **Contract tests:** Enforce schema / required fields.
* **Integration tests:** Run the pipeline end-to-end.
* **Property-based tests:** Fuzz malformed/edge inputs to ensure the code is resilient.

#### 7. ID Repetiton Track (Idempotent)
**If another document with the same ID comes, the code will overwrite the previous record**

---
## 5. Architecture Diagram and Data Flow
![alt text](rough_arch_diag.jpeg)

---
This diagram could have been better, but consider this a rough sketch. I couldn't make a better diagram due to time limitation.
##
## 6. Component: 
`pipeline.py` – Transformation & Validation 🧹

`pipeline.py` contains the **core transformation logic** that turns raw CMS API documents into clean, schema-validated Qdrant documents. Everything that gets embedded and indexed flows through here.

### 6.1.1 Pydantic Schema Models

Two Pydantic models define the target shape of each record:

- **`MetadataModel`**
  - Required:
    - `url: str`
    - `external_id: str`
  - Optional:
    - `title`, `publish_date`, `first_publish_date`, `datetime`, `website`, `thumb`
    - `sections`, `categories`, `tags` (default to empty lists)

- **`QdrantDocument`**
  - `text: str` – cleaned article body
  - `metadata: MetadataModel`

If a document passes these models, it is **guaranteed** to match the Qdrant schema (types + structure).

---

### 6.1.2 Text Extraction & HTML Cleaning

The CMS stores content as nested HTML under `content_elements`. The transformer makes this safe and readable:

- **`clean_text(raw_string)`**
  - Accepts anything (`None`, non-string, HTML).
  - Casts non-strings to `str`.
  - Uses `BeautifulSoup` to strip tags and returns normalized plain text with newline separators.

- **`clean_html(raw_html)`**
  - Returns `(clean_text, tags_found)`:
    - `clean_text`: HTML stripped with `\n` separators.
    - `tags_found`: list of tag names found (for debugging).
  - If input is falsy, returns `("", False)` instead of raising.

- **`get_text_body(raw_doc)`**
  - Iterates `raw_doc.get("content_elements", [])`.
  - Filters for elements with `type == "text"`.
  - Accumulates their `content`, passes through `clean_html`, and builds one unified `text` string.
  - If no usable text is found, the document is later rejected as **“Missing Text”** (hard failure).

---

### 6.1.3 Metadata Extraction Helpers

`DataTransformer` exposes small helper methods to pull each logical field out of the raw JSON:

- **`extract_title`**
  - Reads `headlines["basic"]`, cleaned via `clean_text`.
  - Missing/invalid → `""` (title is optional).

- **`extract_url_general`**
  - Builds a clean absolute URL using:
    - `website_url`
    - `canonical_url`
    - `canonical_website`
  - Prefers:
    1. `https://www.{canonical_website}.com{website_url}` for relative paths.
    2. `website_url` if it already looks like `http(s)://…`.
    3. `canonical_url` if it looks like a full URL.
  - If nothing valid is found → returns `""` and the doc is treated as **Missing URL**.

- **`extract_id`**
  - Reads `_id` and casts to `str`.
  - Missing → `None` → **Missing ID**.

- **Date helpers:**
  - `extract_publish_date`, `extract_first_publish_date`, `extract_datetime`
  - Very conservative: accept only well-formed ISO strings ending in `Z`.
  - Otherwise return `None` (dates are optional).
  - In the original brief, metadata.datetime is described as “typically the same as publish_date but may represent different semantics.” Initially, I treated datetime as identical to publish_date. After clarifying with Jordan, I updated the logic to prioritize the current state of the article rather than only its original publication time. Concretely, I first look for a last_updated_date field and, if present and parsable, I use that as metadata.datetime. If last_updated_date is missing or invalid, I fall back to publish_date. For unchanged articles, this means datetime == publish_date; for articles that have been edited, datetime reflects the most recent update instead.
  - 
- **`extract_website`**
  - Reads `canonical_website` if present.

- **Taxonomy helpers:**
  - `extract_sections` → section names from `taxonomy.sections` (deduped, safe if missing/wrong type).
  - `extract_categories` → top IAB content categories from `taxonomy.categories` (max 5, sorted by score, deduped).
  - `extract_tags` → up to 5 unique slugs from `taxonomy.tags`.
  - All safely fall back to `[]` if the structure isn’t as expected.

- **`extract_thumb`**
  - Looks into `promo_items.basic.additional_properties.resizeUrl` and `canonical_website`.
  - Builds `https://www.{site}.com{resizeUrl}` when possible; otherwise returns `None`.

---

### 6.1.4 Main Orchestrator: `process_document(raw_doc)`

`process_document` is the single entry point for transforming one raw document:

1. **Type Guard**
   - If `raw_doc` is not a `dict`, it logs an error and returns:
     - `result = None`
     - `report = {"status": "SKIPPED", "reason": "Invalid document type"}`

2. **Critical Required Fields**
   - Extract `_id` → `external_id` via `extract_id`
     - Missing → skip with reason `"Missing ID"`.
   - Extract `url` via `extract_url_general`
     - Empty → skip with reason `"Missing URL"`.
   - Build `text` via `get_text_body`
     - Empty/whitespace → skip with reason `"Missing Text"`.

   These three (`_id`, `url`, `text`) are treated as **hard requirements**. If any is missing, the document is not processed further and is sent to the Dead Letter Queue by the caller.

3. **Optional Metadata (Non-blocking)**
   - Attempts to extract `title`, `publish_date`, `first_publish_date`, `datetime`, `website`, `thumb`, `sections`, `categories`, and `tags`.
   - Logs whether each was found.
   - Missing optional fields **do not** cause the document to fail.

4. **Pydantic Validation**
   - Builds a `metadata` dict:
     - Always includes: `url`, `external_id`, `sections`, `categories`, `tags`.
     - Adds other fields only if non-`None`.
   - Validates via:
     ```python
     meta_model = MetadataModel(**metadata)
     doc_model = QdrantDocument(text=text_content, metadata=meta_model)
     ```
   - Any `ValidationError` → document is skipped with reason `"Schema validation failed"`.

5. **Successful Output**
   - Returns:
     - `result = doc_model.model_dump(exclude_none=True)`
     - `report = {"id": external_id, "status": "SUCCESS", "reason": ""}`
   - `exclude_none=True` keeps the final JSON clean (no useless `null`s).

6. **Crash Safety**
   - Any unexpected exception is caught, logged, and reported as:
     - `{"status": "SKIPPED", "reason": f"Crash: {e}"}`

---

### 6.1.5 Batch Mode: Running `pipeline.py` Directly

When `pipeline.py` is executed as a script (`python pipeline.py`):

- It reads `data/raw_customer_api.json`.
- Maintains a `valid_docs_map` and overwrites the file if it has same ID that any older file had (idempotent) **batch output file**.
- For each doc:
  - Calls `process_document(doc)`.
  - On success:
    - Appends the transformed document to `valid_docs`.
  - On failure:
    - Appends a record to the **Dead Letter Queue** file `output/dead_letter_queue.jsonl`:
      - `id`
      - `reason`
      - `raw_doc`

At the end it writes:

- `output/processed_output_updated_2.json` – all successfully transformed documents.
- `output/ingestion_report.csv` – one row per original document (`id`, `status`, `reason`).
- `output/dead_letter_queue.json;` - Dead letter queue

This batch mode is useful for offline runs, debugging, and is also used by the tests to validate the transformation step independently of the API.


## 6.2 Vector Database Service (`vectordb_v3.py`)

`vectordb_v3.py` defines the `VectorDatabase` class, which manages a Qdrant collection and provides upsert and search operations.

### Configuration

- **URL:** `http://localhost:6333`
- Assumes a Qdrant instance is running locally (e.g., via Docker).

### Class: `VectorDatabase`

#### `__init__(self, collection_name: str)`

- Initializes a `QdrantClient` connected to `http://localhost:6333`.
- Stores the target `collection_name` for subsequent operations.

#### `get_or_create_collection(self, vector_size: int = 1536)`

Ensures the collection exists and is configured correctly:

1. Checks if the collection already exists using `collection_exists`.
2. If it does, **deletes it** to ensure a clean, fresh state.
3. Creates a new collection using:
   - `size=vector_size` (default: 1536, matching the embedding size).
   - `distance=Distance.COSINE`.

#### `upsert_documents(self, docs: List[Dict[str, Any]])`

Uploads a batch of documents to Qdrant.

- **Expected document structure:**
  - `doc["vector"]` – the embedding vector.
  - `doc["text"]` – the text content.
  - `doc["metadata"]` – a dictionary of metadata fields.
- **Behavior:**
  1. Iterates over all `docs`.
  2. Skips any document missing a vector or text.
  3. Builds a payload: `payload = { "text": text, "metadata": metadata }`.
  4. Uses the batch loop index i as the point ID.
  5. Creates a Qdrant `PointStruct` with:
     - `id=point_id`
     - `vector` – the embedding list.
     - `payload` – the dictionary containing text and metadata.
  6. Performs a single batch `upsert` operation into the collection.

#### `search(self, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]`

Performs a semantic search over the collection.

- **Input:**
  - `query_vector`: Embedding of the user’s query.
  - `limit`: Max number of results to return (default: 3).
- **Behavior:**
  1. Queries the Qdrant collection for nearest neighbors using cosine similarity.
  2. Iterates over the returned hits.
  3. For each hit, retrieves the stored `payload` (text + metadata).
  4. Adds a `score` field (similarity score) to the result dictionary.
- **Output:** A list of enriched document dictionaries of the form:
  ```json
  {
    "text": "...",
    "metadata": { ... },
    "score": 0.94
  }

## 6.3 Embedding Service (`embedding_v3.py`)

`embedding_v3.py` defines the `EmbeddingModel` class, which acts as a robust client wrapper for the OpenAI API, responsible for converting text into high-dimensional vector representations.

### Configuration

- **API Key:** Requires the `OPENAI_API_KEY` environment variable.
- **Model:** Uses `text-embedding-3-small` (optimized for cost and performance).
- **Dimension:** 1536 dimensions.

### Class: `EmbeddingModel`

#### `__init__(self)`

- Checks for the presence of `OPENAI_API_KEY`.
- **Validation:** Raises a `ValueError` immediately if the key is missing, ensuring the application fails fast rather than at runtime.
- Initializes the standard `OpenAI` client.

#### `generate_embedding(self, text: str) -> List[float]`

Generates a vector embedding for a given text string.

- **Input:** `text` (str).
- **Behavior:**
  1. **Empty Guard:** If the input text is empty or None, immediately returns an empty list `[]`.
  2. **API Call:** Sends the text to OpenAI's `embeddings.create` endpoint.
  3. **Extraction:** Retrieves the vector from the `data[0].embedding` field of the response.
- **Error Handling:**
  - Wraps the external API call in a `try/except` block.
  - If the API fails (network error, rate limit, auth error), it logs the specific error and returns an empty list `[]` instead of crashing the pipeline.
- **Output:** A list of 1536 floats (or `[]` on failure).

## 6.4. API Gateway & Orchestrator (`app.py`)

`app.py` serves as the entry point for the system. It implements a **stateless, robust orchestration layer** using FastAPI. Rather than containing core business logic (like cleaning or database operations), it coordinates the specialized classes (`DataTransformer`, `EmbeddingModel`, `VectorDatabase`) to execute the pipeline stages.

### 6.4.1 Robust Input Handling ("The Bouncer Pattern")

A key architectural decision in `app.py` is "Defensive Ingestion." Instead of using strict Pydantic models for the *input payload* (which would cause a 400 Bad Request for the entire batch if a single item was malformed), the endpoints accept `List[Any]`.

**Logic Flow:**
1.  **Accept Mixed Types:** The API accepts a raw JSON array, even if it contains mixed types (objects, strings, nulls).
2.  **Item-Level Guard:** The code iterates through the list and applies a "Bouncer" check: `isinstance(item, dict)`.
    * **Garbage Data:** Non-dict items (e.g., random strings or nulls) are immediately logged and written to the **Dead Letter Queue**.
    * **Valid Data:** Only dictionary objects are allowed to proceed to the transformation logic.
*Benefit:* A batch of 1,000 documents will not fail just because one item is malformed. The system processes the valid 999 and logs the error for the 1.

### 6.4.2 Endpoint Logic

#### `POST /pipeline/transform`
* **Goal:** Clean raw data and validate schema.
* **Workflow:**
    1.  **Delegation:** Calls `DataTransformer.process_document()` for each valid dictionary.
    2.  **Routing:**
        * **Success:** The clean document is added to the response.
        * **Failure:** The raw document and specific error reason (e.g., "Missing URL") are written to `output/dead_letter_queue.jsonl`.

#### `POST /pipeline/embed`
* **Goal:** Generate vector embeddings for text.
* **Workflow:**
    1.  Filters inputs to ensure they contain a `text` field.
    2.  Calls `EmbeddingModel.generate_embedding(text)`.
    3.  Enriches the document object by adding a `vector` field (e.g., a list of 1536 floats).
    4.  **Fault Tolerance:** Wraps OpenAI calls in a `try/except` block per document, ensuring that one API timeout does not crash the whole batch.

#### `POST /pipeline/index`
* **Goal:** Store documents in Qdrant.
* **Workflow:**
    1.  **Validation:** Filters out any documents that are missing the `vector` field.
    2.  **Dynamic Configuration:** Inspects the first valid vector to determine the required dimension size (e.g., 1536) and calls `VectorDatabase.get_or_create_collection`.
    3.  **Upsert:** Batches the valid documents and sends them to Qdrant via `upsert_documents`.

#### `POST /pipeline/run_full`
* **Goal:** End-to-end processing in a single call.
* **Workflow:** Sequentially chains the logic of Transform → Embed → Index in memory.
    * Useful for quick testing or simple integrations where intermediate states don't need to be inspected by the client.

#### `GET /search`
* **Goal:** Semantic retrieval.
* **Workflow:**
    1.  Accepts a `query` string and limit `k`.
    2.  Converts the query text into a vector using `EmbeddingModel`.
    3.  Performs a nearest-neighbor search using `VectorDatabase.search`.
    4.  Returns the matching documents (text + metadata) and their similarity scores.

## 6.5. Testing Suite 🧪

The project includes a comprehensive test suite (`tests/`) to check data integrity, pipeline robustness, and integration behavior.

### 6.5.1 Contract Tests (`test_contract.py`)

**Goal:** Verify that the transformation logic adheres to the expected Qdrant-style schema.

- **Inputs:** Raw documents from `data/raw_customer_api.json`.
- **Checks:**
  - **Mandatory fields:** `external_id`, `url`, and `text` are present and non-empty.
  - **Data types:** `tags`, `sections`, and `categories` are always lists (even if empty).
  - **Date format:** If present, dates use strict ISO 8601 UTC format.
  - **Pydantic validation:** When available, the output is re-validated using the `QdrantDocument` / `MetadataModel` Pydantic models.

- **Edge cases:**
  - Missing `_id` → document is skipped with reason `"Missing ID"`.
  - Missing text → document is skipped with reason `"Missing Text"`.
  - Missing taxonomy → document is still processed, with list fields defaulting to `[]`.
  - Malformed dates → either the document is skipped or the date is cleaned (never left as arbitrary garbage).

---

### 6.5.2 Integration Tests (`test_integration.py`)

**Goal:** Check the end-to-end flow using real data and the real embedding service.

- **Flow covered:**  
  `raw JSON` → `DataTransformer` → compare with golden record → `EmbeddingModel` → vector checks

- **Key validations:**
  - **Golden record match:** The transformed `text` for a chosen document is compared against a known example in `data/qdrant_format_example.json`. The strings must match exactly.
  - **Live OpenAI call:** Uses the real `EmbeddingModel` to generate an embedding for the transformed text.
  - **Vector verification:** Confirms that:
    - The result is a list.
    - Length is `1536` (for `text-embedding-3-small`).
    - At least some entries are non-zero.

> **Note:** This test requires a valid `OPENAI_API_KEY` in the environment. It is skipped automatically if the key is missing.

---

### 6.5.3 Property-Based Tests (`test_properties.py`)

**Goal:** Use fuzz testing to show that the transformer is stable even when fed “garbage-shaped” data.

- **Tooling:** Uses `hypothesis` to generate many random but JSON-shaped documents.
- **Strategy:**
  - Builds documents with realistic keys (`_id`, `headlines`, `content_elements`, `taxonomy`, dates, etc.) but random values:
    - Empty or `None` IDs,
    - Random text in date fields,
    - Missing or nested taxonomy structures, etc.
  - Each generated document is passed into `DataTransformer.process_document`.

- **Success criteria:**
  1. The transformer does **not crash** on any generated input. If an exception occurs, the test prints the exact failing document and fails.
  2. If a document is accepted (i.e., a result is returned), then:
     - The result is a dictionary.
     - `result["text"]` is a string.
     - `result["metadata"]` is a dictionary.
     - `result["metadata"]["tags"]` is always a list (never `None`).

These tests together cover:
- Contract correctness on real data,
- End-to-end behavior with the real embedding service,
- Robustness to weird but structurally valid inputs.



## 7. Design Decisions
### 7.1 Why Pedantic?

**Decision:** I chose **Pydantic** (`MetadataModel`, `QdrantDocument`) as the strict data validation layer.

* **Why Pydantic?**
    * **Data Validation:** It provides a robust, declarative way to define the expected "shape" of the data. This ensures that only structurally valid records enter the embedding pipeline.
    * **Performance:** Pydantic V2 is **highly optimized**, offering significant speed advantages over standard Python validation methods, which is critical when processing millions of records.
    * **Schema Evolution & Compatibility:**
      * **Backward Compatibility:** By using `Optional` types and default values (e.g., `default_factory=list`), the schema natively handles legacy data that might be missing newer fields.
      * **Forward Compatibility:** The model is configured to ignore unknown fields by default. If the upstream API evolves and adds new fields, the pipeline will not crash; it will simply ignore the extra data.
    * **Result:** This native support for both directions allows the schema to evolve over time without requiring breaking code changes or complex migration scripts.


### 7.2 Vector Database: Qdrant


**Decision:** I chose **Qdrant** (Dockerized locally) as the vector store.

* **Why Qdrant:**
    * **Exceptional Speed and Performance:** Built in Rust, Qdrant is highly performant. It achieves millisecond-level retrieval times through efficient **HNSW (Hierarchical Navigable Small World)** indexing. HNSW allows search to grow roughly logarithmically with the number of vectors, making large datasets searchable in milliseconds.
    * **Real-Time Analytics:** Qdrant supports real-time multi-modal data processing and real-time decision-making.
    * **Memory and Cost Efficiency:** The platform offers advanced compression techniques such as scalar, binary, and product quantization. Binary quantization dramatically reduces the memory footprint of vectors, offering memory usage cuts by up to 32x.
    * **Cloud-Native Scalability:** Qdrant supports both vertical and horizontal scaling with features like sharding and replication for high availability.
    * **Unified Storage:** Qdrant natively stores a vector and the arbitrary JSON payload (metadata) per point. This makes it a **Single Source of Truth**, eliminating the need for a separate relational database for metadata.
    * **Open Source Model:** The open-source nature helps avoid vendor lock-in, fosters a strong community, and allows for transparency and customizability.
  
### 7.3 Why FastAPI?

**Decision:** I chose FastAPI as the web framework for the API Gateway and Orchestrator (`app.py`).

**Why FastAPI?**

- **High Performance and Concurrency**  
  FastAPI is built on ASGI (using Starlette and Uvicorn), with native `async/await` support.

  **Benefit:** In an I/O-bound pipeline (waiting on OpenAI and Qdrant network calls), async endpoints let the service handle many requests in parallel instead of blocking on each one.

- **First-Class Pydantic Integration**  
  FastAPI uses Pydantic under the hood for request/response models.

  **Benefit:** Request bodies are automatically validated and parsed into typed objects, and responses are serialized back to JSON. This reduces boilerplate and keeps data contracts consistent with the internal `MetadataModel` / `QdrantDocument` definitions.

- **Developer Experience & Architecture**
  - **Automatic Docs:** FastAPI exposes interactive API docs (Swagger UI / ReDoc) via OpenAPI. This makes it easy for anyone to inspect and try the endpoints from a running container.
  - **API-First Design:** FastAPI is built for lightweight, stateless services. That matches the goal of keeping `app.py` as a thin orchestration layer that’s easy to containerize and scale horizontally.

**Conclusion:** FastAPI gives a good balance of performance, clean type-driven design, and low boilerplate, making it a natural fit for this kind of high-throughput, JSON-only ingestion and search service.

### 7.4 Code Modularity & Separation of Concerns

**Decision:** The project is engineered around high modularity, with code split into specialized files (e.g., `pipeline.py` for transformation, `embedding_v3.py` for embedding API calls, `vectordb_v3.py` for Qdrant, and `app.py` for routing/orchestration).

**Why this architecture?**

- **Testability:**  
  Each module can be tested in isolation. For example, the text cleaning and schema logic in `pipeline.py` can be unit-tested without spinning up FastAPI or mocking the Qdrant client. This keeps tests focused and makes regressions easier to catch and localize.

- **Flexibility & Reusability:**  
  The core business logic is decoupled from the delivery mechanism:
  - `DataTransformer` can be run from a CLI script for batch backfilling or invoked via the API for real-time ingestion, with no code duplication.
  - If the embedding provider changes (e.g., from OpenAI to a local Hugging Face model), only `embedding_v3.py` needs to be updated. `app.py` and `vectordb_v3.py` do not change.

- **Maintainability:**  
  The separation makes it easier to onboard new developers and debug issues:
  - Index / search problems are confined to `vectordb_v3.py`.
  - Data validation / transformation issues live in `pipeline.py`.
  - Request/response and routing concerns are isolated in `app.py`.
  This clear ownership reduces cognitive load when diagnosing bugs.

- **Statelessness:**  
  The orchestration layer (`app.py`) is intentionally kept stateless. All persistent concerns (vectors, documents) are delegated to external systems (Qdrant, storage). This design is a natural fit for cloud-native deployments where horizontal scaling behind a load balancer is expected.

Overall, the modular structure ensures that each part of the system does one job well, and changes in one layer do not ripple unnecessarily through the rest of the codebase.

### 7.5 Logging, Dead Letter Queue & Reporting

**Decision:** I built explicit observability into the pipeline using three layers:
1. Structured logging (`output/pipeline.log`)
2. A dead letter queue (`output/dead_letter_queue.jsonl`)
3. A summary CSV report (`output/ingestion_report.csv`)

**Structured logging**

- The transformer uses Python’s `logging` module with both:
  - A file handler → `output/pipeline.log`
  - A console handler → for local runs / debugging
- For each document, the log records:
  - Document ID (`_id`)
  - Whether URL/text/title/dates/taxonomy/thumbnail were found
  - Final status (`SUCCESS` vs reason for skip/failure)
- This gives a chronological “story” of the ingestion run and makes it easy to answer questions like:
  - *“Why did this specific article not show up in the index?”*
  - *“Are we systematically missing dates for a particular source?”*

**Dead Letter Queue (DLQ)**

- Any document that fails critical checks (missing `_id`, missing URL, empty text, schema validation failure, etc.) is not silently dropped.
- Instead, it is written to a JSONL file:
  - File: `output/dead_letter_queue.jsonl`
  - Each line contains:
    - `id`: original `_id`
    - `reason`: e.g. `"Missing URL"`, `"Missing Text"`, `"Schema validation failed"`
    - `raw_doc`: the full original payload
- This design has two benefits:
  - Operations: Failed records can be re-ingested later after fixing upstream issues, without re-running the full batch.
  - Debugging: I can quickly inspect “bad” documents to see if the problem is in the source data or in the transformation logic.

**Ingestion summary report**

- In addition to logs and DLQ, the pipeline writes a CSV summary:
  - File: `output/ingestion_report.csv`
  - Columns: `id`, `status`, `reason`
- This gives a compact bird’s-eye view:
  - How many documents succeeded vs failed
  - Top failure reasons (useful for dashboards or quick Excel analysis)
- In the main script, I also print aggregate stats at the end:
  - Total input count
  - Unique successful documents
  - Number of duplicates merged (same `external_id`)
  - Number of failures (sent to DLQ)

Together, the logging + DLQ + CSV report make the pipeline observable and debuggable in production: no silent drops, clear failure modes, and an easy path to replay or inspect problematic records.


### 7.6 Why OpenAI Embeddings (and not Hugging Face / local models)?

**Decision:** I use OpenAI’s `text-embedding-3-small` as the single embedding backend, wrapped in a small `EmbeddingModel` class.

**Why this choice?**

- **Good quality without extra infra**
  - For this use case (news articles + metadata), I need strong semantic search, not a custom fine-tuned model.
  - `text-embedding-3-small` gives a solid retrieval baseline with a **fixed, documented dimension (1536)**, which I hard-code into the Qdrant collection configuration and tests.
  - I do *not* have to manage GPUs, model loading, batching, or a separate model-serving stack.

- **Cost vs. complexity trade-off**
  - Hugging Face / local sentence-transformer models are **“free”** per call, but they require:
    - A host with enough CPU/GPU,
    - A server process (FastAPI/gRPC/etc.) for inference,
    - Tuning batch sizes, warm-up, and concurrency.
  - For this project, Paying a small per-request cost to OpenAI is simpler and more predictable than running my own embedding service.



- **Future flexibility**
  - If later the team decides to move to a Hugging Face model or an on-prem encoder, I only need to change the implementation inside `EmbeddingModel`.  
    - The FastAPI endpoints, Qdrant integration, and tests do not need to change because they depend on the *contract* (“give me a vector of length 1536”), not on OpenAI specifically.

In short, OpenAI embeddings let me keep the pipeline simple, deterministic, and easy to swap later, while avoiding the operational overhead of self-hosting a transformer model for this assessment.

### 7.7 Why Render for Deployment?

**Decision:** I deployed the FastAPI API on **Render** because it was the simplest way to get a real, public endpoint running from my Docker image.

**Why Render (in simple terms):**

- **Very easy from Docker → URL**  
  The app is already in a Docker image, and Render can run that directly. I don’t need to set up nginx, HTTPS, or any server config by hand.

- **Automatic deploys from GitHub**  
  Render connects to the repo and rebuilds/redeploys on push. That’s enough for this project without adding a full CI/CD pipeline.

- **Built-in environment variables**  
  Secrets like `OPENAI_API_KEY` and Qdrant connection details are set as env vars in the Render dashboard, so I don’t hard-code anything.

In short, Render was the most straightforward way to show the API running “for real” without turning this into a full DevOps project.

## 8 Trade-offs, Limitations & Future Work

This implementation is intentionally pragmatic: it is production-shaped, but not yet production-hardened. Below are the main trade-offs behind each design choice and what I would change if this needed to serve millions of documents and high QPS.

#### Pydantic (Validation Layer)

**Limitations / Trade-offs**

- Pydantic validation adds CPU overhead when every document is hydrated into a model. At very high throughput, this becomes a non-trivial cost.
- The schema is strict by design. If upstream teams start sending partially valid data with new fields or slightly different shapes, those records can be rejected until the schema is updated.

**Future Work**

- Introduce a “lightweight mode” that validates only critical fields (IDs, URLs, dates) on hot paths, and reserves full Pydantic validation for backfills or offline QA.
- Version the schema (v1, v2, …) and add explicit migration logic between versions instead of relying purely on Optional fields.

---

#### Qdrant (Vector Database)

**Limitations / Trade-offs**

- Running Qdrant ourselves means we own tuning and operations: HNSW parameters, shard/replica configuration, backup/restore, and monitoring.
- For truly massive scales (billions of vectors + strict SLAs), index tuning and hardware sizing become critical and require specialist knowledge.
- Right now the setup assumes a single Qdrant instance; there is no cross-region redundancy or automated failover.

**Future Work**

- Move to **Qdrant Cloud** or a managed deployment with autoscaling, backups, and monitoring built-in.
- Experiment with HNSW parameters (M, ef_construct, ef_search) and quantization strategies for different workloads (read-heavy vs write-heavy).
- Add hybrid retrieval (e.g., vector + metadata filters) and benchmark latency/recall to tune the schema and indexing strategy.

---

#### FastAPI (API Gateway & Orchestrator)

**Limitations / Trade-offs**

- Internal code is mostly synchronous today; I am not fully exploiting FastAPI’s async capabilities for concurrent OpenAI / Qdrant calls.
- Long-running batch operations (e.g., large re-indexing jobs) are not yet pushed to background workers; they’d block a worker if exposed directly via API.
- There is no streaming interface yet (e.g., for incremental ingestion progress or long-running search jobs).

**Future Work**

- Make the I/O-heavy parts (OpenAI + Qdrant calls) truly `async` and/or batch them to reduce per-request latency.
- Introduce a background task queue (e.g., Celery, RQ, Dramatiq) for large ingestion jobs, with FastAPI only enqueuing work and returning job IDs.
- Add rate limiting and backoff policies at the API layer to shield downstream services from overload.

---

#### Code Modularity & Separation of Concerns

**Limitations / Trade-offs**

- For a small project, having multiple modules (pipeline.py, embedding_v3.py, vectordb_v3.py, app.py) can feel “heavier” than a single script.
- There is no formal plugin system yet; swapping components still requires editing Python files and redeploying.

**Future Work**

- Package the core logic (DataTransformer, EmbeddingModel, VectorDatabase) as an installable library so multiple services can share it.
- Introduce simple interfaces / protocols (e.g., `EmbeddingBackend`, `VectorStore`) and register implementations via config, not code changes.
- Add stricter type-checking (mypy) and linting to keep module boundaries clean as the codebase grows.

---

#### Logging, Dead Letter Queue & Reporting

**Limitations / Trade-offs**

- Logs and DLQ are currently file-based (`pipeline.log`, `dead_letter_queue.jsonl`), which is fine for a single instance but not ideal for a multi-node cluster.
- JSONL DLQ is great for inspection, but not ideal for high-volume replay and backpressure handling.
- The CSV report is static; it’s not plugged into any dashboard or alerting system.

**Future Work**

- Move logging to a centralized system (e.g., ELK stack, OpenTelemetry + Grafana), with structured JSON logs and trace IDs.
- Replace the file-based DLQ with a queue or topic (e.g., Kafka, SQS, Pub/Sub) so failed records can be replayed, inspected, or routed to separate remediation pipelines.
- Generate ingestion metrics (success/fail counts, top failure reasons) and export them as Prometheus metrics for real-time monitoring and alerting.

---

#### OpenAI Embeddings

**Limitations / Trade-offs**

- All text sent for embeddings goes to an external provider, which may not be acceptable for datasets with strict privacy, PII, or compliance constraints.
- At very high volumes, API costs and rate limits become a real bottleneck.
- Dependence on a single external embedding provider introduces vendor lock-in (model changes, pricing changes, outages).

**Future Work**

- Implement a pluggable embedding backend with at least two options:
  - OpenAI (managed, external)
  - Local Hugging Face model (self-hosted, internal)
- Add caching for embeddings keyed by a stable hash of the text to avoid recomputing vectors for unchanged content.
- Add a migration path for changing embedding dimensions (e.g., storing both “v1” and “v2” embeddings during a transition period).

---

#### Render Deployment

**Limitations / Trade-offs**

- Render is optimized for convenience, not for ultra-low latency or full control over networking and autoscaling.
- Services can experience **cold starts** after periods of inactivity, which adds initial latency.
- Infrastructure is somewhat opaque; deep tuning of autoscaling, regional placement, and node types is limited compared to raw EC2/Kubernetes.

**Future Work**

- Move to a more configurable setup (e.g., AWS ECS/EKS or raw EC2 with a load balancer) once traffic, SLA, and cost profiles are clearer.
- Add a full CI/CD pipeline (GitHub Actions, etc.) with automated tests, security scans, and canary deployments before rolling out changes.

---

#### No PySpark / Distributed Compute (Yet)

**Limitations / Trade-offs**

- The current pipeline runs on a single machine and assumes the dataset fits in memory / local disk.
- For very large corpora (hundreds of millions of articles), single-node processing would either be too slow or exceed resource limits.

**Future Work**

- Use Spark (or Dask / Ray) for large-scale backfills and offline re-indexing.
- Keep the FastAPI + Qdrant stack as the online serving layer, while distributed jobs periodically push new/updated embeddings into Qdrant.

---

#### Limited Use of AsyncIO

**Limitations / Trade-offs**

- While FastAPI supports async endpoints, much of the underlying logic is synchronous, so the event loop is not fully utilized.
- Under heavy concurrent load, synchronous calls to external services can reduce throughput and increase tail latency.

**Future Work**

- Refactor the integration with OpenAI and Qdrant to use async clients where possible.
- Introduce concurrency controls (semaphores, connection pools) and structured retries with exponential backoff to make the system more resilient under load.
#### Single API Instance Today (No Load Balancer Yet)

Right now, the FastAPI service runs as a **single instance** (one container on Render). That’s fine for this assessment and light traffic, but it has some clear limitations:

- If that one instance goes down, the API is temporarily unavailable.
- Throughput is bounded by the CPU/RAM of a single container.
- Deployments are simple, but there is no blue/green or canary rollout story.

This is largely an **intentional simplification**: the app layer is fully stateless, so it’s easy to reason about and easy to demo on a single URL.

**Future Work: Multiple Replicas + Load Balancing**

In a production setting with higher traffic, the natural next step would be to:

- Run **multiple FastAPI replicas**:
  - For example, several containers behind a load balancer (Render autoscaling, AWS ALB + ECS/EKS, or Nginx/Envoy in front of a Kubernetes cluster).
  - Each replica would be identical and stateless; all persistent data stays in Qdrant and external storage.

- Add **health checks and rolling deploys**:
  - Load balancer pings `/health` or `/ready` on each instance.
  - During deploys, new versions are brought up and checked before traffic is shifted over (blue/green or rolling updates).

- Use **autoscaling policies**:
  - Scale out when latency or CPU crosses a threshold.
  - Scale back in when load drops, to save cost.

Because the current design already keeps `app.py` stateless and delegates all storage to Qdrant and external services, moving to a multi-instance, load-balanced setup is mostly an **infrastructure change**, not an application rewrite.

## 9 Honest and basic improvements
### 9.1 The pipeline fails on invalid json structure, this is something that needs to be improved on.
### 9.2 The app.py doesn't write to disk.
### 9.3 Date handling can be done better
### 9.4 The testing could have covered more cases
### 9.5 I could have implemented dashboards from logs that I made using pipeline
### 9.6 Parallelization could have been done
### 9.7 Diagram could have been better
### 9.8 I could have clarified date handling much earlier