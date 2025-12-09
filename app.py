import logging
import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException

# Import logic from your new modules
# NOTE: Make sure your file names match these imports (pipeline.py, embeddings_V2.py, vector_db_V2.py)
from pipeline import execute_transformation_step
from embeddings_V2 import execute_embedding_step
from vector_db_V2 import execute_indexing_step, execute_search, SearchResult

# Config
INPUT_RAW_FILE = "raw_customer_api.json"
OUTPUT_PROCESSED_FILE = "processed_output.json"
OUTPUT_EMBEDDED_FILE = "processed_output_with_embeddings.json"
COLLECTION_NAME = "capitol_assessment"

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Modular RAG Pipeline")

# ==============================================================================
# INDIVIDUAL STEP ENDPOINTS
# ==============================================================================

@app.post("/pipeline/transform", summary="Step 1: Transform Only")
def api_transform_data():
    """Step 1: Cleans raw JSON and saves to processed_output.json"""
    try:
        docs = execute_transformation_step(INPUT_RAW_FILE, OUTPUT_PROCESSED_FILE)
        return {
            "status": "success",
            "message": f"Successfully transformed {len(docs)} documents.",
            "output_file": OUTPUT_PROCESSED_FILE
        }
    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/embed", summary="Step 2: Embed Only")
def api_embed_documents():
    """Step 2: Reads cleaned JSON, adds embeddings, saves to processed_output_with_embeddings.json"""
    try:
        embedded_docs = execute_embedding_step(OUTPUT_PROCESSED_FILE, OUTPUT_EMBEDDED_FILE)
        return {
            "status": "success",
            "message": f"Successfully embedded {len(embedded_docs)} documents.",
            "output_file": OUTPUT_EMBEDDED_FILE
        }
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/index", summary="Step 3: Index Only")
def api_index_documents():
    """Step 3: Reads embedded JSON and uploads to Vector DB"""
    try:
        execute_indexing_step(OUTPUT_EMBEDDED_FILE, COLLECTION_NAME)
        return {
            "status": "success",
            "message": f"Successfully created and indexed collection '{COLLECTION_NAME}'."
        }
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# FULL PIPELINE & SEARCH
# ==============================================================================

@app.post("/pipeline/run_full", summary="Run All Steps")
def api_run_full_pipeline():
    """Runs Transform -> Embed -> Index in sequence."""
    try:
        # 1. Transform
        docs = execute_transformation_step(INPUT_RAW_FILE, OUTPUT_PROCESSED_FILE)
        logger.info(f"Transformed {len(docs)} documents.")
        
        # 2. Embed
        embedded_docs = execute_embedding_step(OUTPUT_PROCESSED_FILE, OUTPUT_EMBEDDED_FILE)
        logger.info(f"Embedded {len(embedded_docs)} documents.")
        
        # 3. Index
        execute_indexing_step(OUTPUT_EMBEDDED_FILE, COLLECTION_NAME)
        logger.info("Indexing complete.")
        
        return {
            "status": "success", 
            "message": "Full pipeline complete.",
            "docs_processed": len(embedded_docs)
        }
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search", response_model=List[SearchResult], summary="Search Documents")
def api_search(query: str, k: int = 3):
    """Semantic Search endpoint."""
    try:
        return execute_search(query, COLLECTION_NAME, k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Using port 8090 as you requested
    uvicorn.run(app, host="127.0.0.1", port=8050)