import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- IMPORTS ---
# This code assumes you have these classes defined in your other files:
# 1. pipeline.py      -> DataTransformer
# 2. embeddings_V2.py -> EmbeddingModel
# 3. vector_db_V2.py  -> VectorDatabase
from pipeline import DataTransformer
from embedding_v3 import EmbeddingModel 
from vector_db_V2 import VectorDatabase 

# --- CONFIG & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CapitolPipeline")

app = FastAPI(title="Modular RAG Pipeline (Stateless)")
COLLECTION_NAME = "capitol_assessment"

# --- ROOT ENDPOINT (To avoid 404s) ---
@app.get("/")
def read_root():
    return {
        "status": "System is live!",
        "docs_url": "/docs",
        "endpoints": [
            "/pipeline/transform",
            "/pipeline/embed",
            "/pipeline/index",
            "/pipeline/run_full",
            "/search"
        ]
    }

# ==============================================================================
# 1. TRANSFORM ENDPOINT (Raw JSON -> Clean JSON)
# ==============================================================================
@app.post("/pipeline/transform", summary="Step 1: Transform Raw Data")
def api_transform_data(raw_data: List[Dict[str, Any]]):
    """
    Accepts a list of raw customer documents.
    Returns a list of clean, Qdrant-compliant documents.
    """
    try:
        transformer = DataTransformer()
        valid_docs = []
        skipped_count = 0
        seen_ids = set()

        logger.info(f"Transforming {len(raw_data)} documents...")

        for doc in raw_data:
            doc_id = doc.get('_id')
            
            # Simple Deduplication
            if doc_id in seen_ids: 
                skipped_count += 1
                continue
            if doc_id: seen_ids.add(doc_id)

            # Process document
            result, report = transformer.process_document(doc)
            
            if result:
                valid_docs.append(result)
            else:
                skipped_count += 1

        return {
            "status": "success",
            "processed_count": len(valid_docs),
            "skipped_count": skipped_count,
            "documents": valid_docs
        }
    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# 2. EMBED ENDPOINT (Clean JSON -> Vectorized JSON)
# ==============================================================================
@app.post("/pipeline/embed", summary="Step 2: Generate Embeddings")
def api_embed_documents(processed_docs: List[Dict[str, Any]]):
    """
    Accepts list of cleaned documents.
    Returns documents with a new 'vector' field added.
    """
    try:
        embedder = EmbeddingModel()
        embedded_docs = []

        logger.info(f"Embedding {len(processed_docs)} documents...")

        for doc in processed_docs:
            text = doc.get("text", "")
            if not text:
                continue
            
            # Generate Vector
            vector = embedder.generate_embedding(text)
            doc["vector"] = vector
            embedded_docs.append(doc)

        return {
            "status": "success",
            "embedded_count": len(embedded_docs),
            "documents": embedded_docs
        }
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# 3. INDEX ENDPOINT (Vectorized JSON -> Database)
# ==============================================================================
@app.post("/pipeline/index", summary="Step 3: Index to Qdrant")
def api_index_documents(embedded_docs: List[Dict[str, Any]]):
    """
    Accepts list of documents with vectors.
    Uploads them to Qdrant.
    """
    try:
        if not embedded_docs:
            return {"status": "warning", "message": "No documents provided to index."}

        # Initialize DB
        vector_db = VectorDatabase(collection_name=COLLECTION_NAME)
        
        # Ensure collection exists (Check vector size dynamically)
        sample_vector = embedded_docs[0].get("vector")
        vector_size = len(sample_vector) if sample_vector else 384
        vector_db.get_or_create_collection(vector_size=vector_size)

        logger.info(f"Indexing {len(embedded_docs)} documents...")
        
        # Batch Upsert
        vector_db.upsert_documents(embedded_docs)

        return {
            "status": "success",
            "message": f"Successfully indexed {len(embedded_docs)} documents."
        }
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# 4. FULL PIPELINE (Raw JSON -> Database)
# ==============================================================================
@app.post("/pipeline/run_full", summary="Run Full Pipeline")
def api_run_full_pipeline(raw_data: List[Dict[str, Any]]):
    """
    Takes Raw JSON input, runs Transform -> Embed -> Index automatically.
    """
    try:
        # 1. Transform
        transformer = DataTransformer()
        clean_docs = []
        seen_ids = set()
        for doc in raw_data:
            doc_id = doc.get('_id')
            if doc_id in seen_ids: continue
            if doc_id: seen_ids.add(doc_id)
            
            res, _ = transformer.process_document(doc)
            if res: clean_docs.append(res)
        
        if not clean_docs:
            return {"status": "warning", "message": "No valid documents found."}

        # 2. Embed
        embedder = EmbeddingModel()
        embedded_docs = []
        for doc in clean_docs:
            doc["vector"] = embedder.generate_embedding(doc.get("text", ""))
            embedded_docs.append(doc)

        # 3. Index
        vector_db = VectorDatabase(collection_name=COLLECTION_NAME)
        vector_db.get_or_create_collection(vector_size=len(embedded_docs[0]["vector"]))
        vector_db.upsert_documents(embedded_docs)

        return {
            "status": "success",
            "message": "Full pipeline completed successfully.",
            "processed_count": len(clean_docs),
            "indexed_count": len(embedded_docs)
        }
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# SEARCH ENDPOINT
# ==============================================================================
@app.get("/search", summary="Semantic Search")
def api_search(query: str, k: int = 3):
    try:
        embedder = EmbeddingModel()
        query_vector = embedder.generate_embedding(query)
        
        vector_db = VectorDatabase(collection_name=COLLECTION_NAME)
        results = vector_db.search(query_vector, limit=k)
        
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)