#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os

os.environ["OPENAI_API_KEY"] = "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"


# In[4]:


# embedding_pipeline.py

import json
import os
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
INPUT_FILE = "processed_output.json"
OUTPUT_FILE = "processed_output_with_embeddings.json"


def main():
    """
    Very simple embedding step:

    1. Read transformed docs from INPUT_FILE
    2. Call OpenAI embeddings once with all `text` values
    3. Attach `embedding` to each doc
    4. Save to OUTPUT_FILE
    """
    # ---- 1. Setup client (same as before, just no hard-coded key) ----
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)

    # ---- 2. Load documents from pipeline output ----
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    if not isinstance(docs, list):
        raise ValueError("Expected a list of documents in processed_output.json")

    texts = [doc["text"] for doc in docs]
    print(f"Embedding {len(texts)} documents with model={EMBEDDING_MODEL}...")

    # ---- 3. Call embedding API in one batch (same as your logic) ----
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = [item.embedding for item in response.data]
    if len(embeddings) != len(docs):
        raise RuntimeError("Number of embeddings does not match number of documents")

    dim = len(embeddings[0]) if embeddings else 0
    print(f"Got embeddings (dimension={dim})")

    # ---- 4. Attach embeddings and save ----
    for doc, emb in zip(docs, embeddings):
        doc["embedding"] = emb

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print(f"Saved documents with embeddings to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


# In[ ]:


import os
import json
from typing import List, Dict
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"

def execute_embedding_step(input_file: str, output_file: str) -> List[Dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: raise RuntimeError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    try:
        with open(input_file, "r", encoding="utf-8") as f: docs = json.load(f)
    except FileNotFoundError: return []
    
    if not docs: return []
    texts = [doc["text"] for doc in docs if "text" in doc]
    if not texts: return docs

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    embeddings = [item.embedding for item in response.data]

    embedded_docs = []
    for doc, emb in zip(docs, embeddings):
        doc["embedding"] = emb
        embedded_docs.append(doc)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embedded_docs, f, indent=2, ensure_ascii=False)
    
    return embedded_docs

