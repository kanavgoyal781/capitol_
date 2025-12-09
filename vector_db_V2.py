#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os

os.environ["OPENAI_API_KEY"] = "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"


# In[9]:


# vector_db.py
"""
Very small helper for:
1. Loading docs with embeddings from JSON
2. Indexing them into local Qdrant
3. Running one demo semantic search
"""

import json
import os

from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI

# ---- Simple config ----
COLLECTION_NAME = "capitol_assessment"
VECTOR_SIZE = 1536                  # text-embedding-3-small
EMBEDDING_MODEL = "text-embedding-3-small"
INPUT_FILE = "processed_output_with_embeddings.json"


def get_qdrant_client():
    """Connect to local Qdrant (run via Docker)."""
    return QdrantClient(host="localhost", port=6333)


def load_docs_with_embeddings():
    """
    Load documents from the pipeline+embedding output.

    Expected per-doc:
        {
          "text": "...",
          "metadata": {...},
          "embedding": [floats... length 1536]
        }
    """
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    if not isinstance(docs, list):
        raise ValueError(f"Expected a list of documents in {INPUT_FILE}")

    return docs


def index_documents_to_qdrant(docs):
    """
    Create/reset the Qdrant collection and upload all docs as points.

    Idempotent: safe to re-run, because it recreates the collection.
    """
    client = get_qdrant_client()

    # Reset collection → satisfies "idempotent processing" requirement
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )

    points = []
    for i, doc in enumerate(docs):
        emb = doc.get("embedding")
        text = doc.get("text")
        metadata = doc.get("metadata", {})

        # Minimal validation so we don't send garbage to Qdrant
        if not isinstance(emb, list):
            continue
        if len(emb) != VECTOR_SIZE:
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(metadata, dict):
            metadata = {}

        payload = {"text": text, **metadata}

        points.append(
            models.PointStruct(
                id=i,
                vector=emb,
                payload=payload,
            )
        )

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"✅ Uploaded {len(points)} points to collection '{COLLECTION_NAME}'")


def run_example_search(query_text="Who is Zack Wheeler?", top_k=3):
    """
    Simple semantic search demo:
    - Builds query embedding with OpenAI
    - Queries Qdrant
    - Prints top_k results.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    openai_client = OpenAI(api_key=api_key)
    qdrant_client = get_qdrant_client()

    print(f"🔎 Generating embedding for: '{query_text}'...")
    query_vector = openai_client.embeddings.create(
        input=[query_text],
        model=EMBEDDING_MODEL,
    ).data[0].embedding

    print("🚀 Searching database...")
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )

    hits = results.points or []

    print("\n--- Search Results ---")
    for i, hit in enumerate(hits):
        payload = hit.payload or {}
        title = payload.get("title", "No Title")
        text = payload.get("text", "") or ""
        snippet = text[:1500]

        print(f"\nResult #{i+1} (Score: {hit.score:.4f})")
        print(f"Title: {title}")
        print(f"Snippet: {snippet}...")  


if __name__ == "__main__":
    print(f"📥 Loading documents from {INPUT_FILE} ...")
    docs = load_docs_with_embeddings()
    print(f"   → Loaded {len(docs)} docs")

    print("📦 Indexing into Qdrant ...")
    index_documents_to_qdrant(docs)

    print("\n🔍 Running demo search...")
    run_example_search()


# In[10]:


import os
import json
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydantic import BaseModel
from openai import OpenAI

# --- Config ---
VECTOR_SIZE = 1536
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "capitol_assessment"

# --- Shared Pydantic Model for Search Results ---
class SearchResult(BaseModel):
    score: float
    title: str
    snippet: str
    url: str

def get_qdrant_client():
    """
    Intelligent connection:
    - If QDRANT_HOST is set (Render/Cloud), use that.
    - Otherwise, default to localhost (Docker).
    """
    host = os.getenv("QDRANT_HOST")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if host:
        return QdrantClient(url=host, api_key=api_key)
    return QdrantClient(host="localhost", port=6333)

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)

# --- Function called by app.py for Step 3 ---
def execute_indexing_step(input_file: str, collection_name: str):
    client = get_qdrant_client()
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            docs = json.load(f)
    except FileNotFoundError:
        print(f"File {input_file} not found.")
        return
    
    if not docs:
        print("No docs to index.")
        return

    # Idempotent: Recreate collection to ensure fresh state
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE, 
            distance=models.Distance.COSINE
        ),
    )

    points = []
    for i, doc in enumerate(docs):
        emb = doc.get("embedding")
        text = doc.get("text")
        metadata = doc.get("metadata", {})
        
        # Validations
        if not isinstance(emb, list) or len(emb) != VECTOR_SIZE: continue
        if not text: continue
        
        payload = {"text": text, **metadata}
        
        points.append(models.PointStruct(
            id=i,
            vector=emb,
            payload=payload
        ))
    
    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Indexed {len(points)} documents.")

# --- Function called by app.py for Search ---
def execute_search(query_text: str, collection_name: str, top_k: int = 3) -> List[SearchResult]:
    q_client = get_qdrant_client()
    o_client = get_openai_client()
    
    # 1. Generate Query Embedding
    response = o_client.embeddings.create(input=[query_text], model=EMBEDDING_MODEL)
    query_vector = response.data[0].embedding
    
    # 2. Search Qdrant
    if not q_client.collection_exists(collection_name):
        return []
        
    results = q_client.query_points(
        collection_name=collection_name, 
        query=query_vector, 
        limit=top_k
    )
    
    # 3. Format Results
    formatted_results = []
    for hit in results.points or []:
        payload = hit.payload or {}
        formatted_results.append(SearchResult(
            score=hit.score,
            title=payload.get("title", "No Title"),
            snippet=payload.get("text", "")[:500] + "...", # Truncate snippet
            url=payload.get("url", "No URL")
        ))
        
    return formatted_results

# --- Local Testing Block ---
if __name__ == "__main__":
    # You can run this file directly to test just the DB part
    print("Test run...")
    # execute_indexing_step("processed_output_with_embeddings.json", COLLECTION_NAME)
    # res = execute_search("Zack Wheeler", COLLECTION_NAME)
    # print(res)


# In[ ]:




