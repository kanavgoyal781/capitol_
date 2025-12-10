import pytest
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import DataTransformer
# We import QdrantDocument for strict validation if available
try:
    from pipeline import QdrantDocument
except ImportError:
    QdrantDocument = None

# Import the REAL Embedding Service
try:
    from embedding_v3 import EmbeddingModel
except ImportError:
    EmbeddingModel = None

def load_datasets():
    try:
        with open("data/raw_customer_api.json", "r") as f:
            raw = json.load(f)
        with open("data/qdrant_format_example.json", "r") as f:
            expected = json.load(f)
        return raw, expected
    except FileNotFoundError:
        return [], []

RAW_DATA, EXPECTED_DATA = load_datasets()

@pytest.mark.skipif(not EmbeddingModel, reason="embedding_v3.py not found")
def test_full_ingestion_integration_real_api():
    """
    REAL Integration Test:
    1. EXTRACT: Takes a real document from raw_customer_api.json.
    2. TRANSFORM: Runs DataTransformer pipeline.
    3. VALIDATE: Matches text against Golden Record (by ID).
    4. EMBED: Calls REAL OpenAI API to generate a vector.
    5. VERIFY: Checks vector dimensions (1536).
    """
    
    # --- LOGGING & CHECKS ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ CRITICAL ERROR: 'OPENAI_API_KEY' not found in environment.")
        print("   You must set it to run integration tests.")
        print("   Command: export OPENAI_API_KEY='sk-...'")
        pytest.skip("Skipping: API Key missing")
        return

    print("\n🚀 Starting Real Integration Test (OpenAI API)...")

    # --- SETUP ---
    transformer = DataTransformer()
    embedder = EmbeddingModel() # Real Client

    # --- ROBUST DATA MATCHING (Handle Shuffling) ---
    # Convert list of golden records into a Dictionary keyed by external_id
    golden_map = {
        doc.get('metadata', {}).get('external_id'): doc 
        for doc in EXPECTED_DATA
    }
    
    test_doc = None
    golden_record = None
    
    # Find the first document in RAW that also exists in GOLDEN
    for doc in RAW_DATA:
        doc_id = doc.get('_id')
        if doc_id in golden_map:
            test_doc = doc
            golden_record = golden_map[doc_id]
            break
            
    if not test_doc:
        pytest.fail("Could not find any matching documents between raw and golden datasets.")

    print(f"   ℹ️  Matched Document ID: {test_doc.get('_id')}")

    # --- STEP 1: TRANSFORMATION ---
    print("   🔄 Step 1: Running Pipeline Transformation...")
    result_dict, report = transformer.process_document(test_doc)
    assert result_dict is not None, f"Pipeline failed: {report}"

    # --- STEP 2: SCHEMA VALIDATION (Pydantic) ---
    if QdrantDocument:
        try:
            # Strict validation using your own pipeline's definition
            validated_doc = QdrantDocument(**result_dict)
            print("   ✅ Step 2: Output strictly adheres to Pydantic Schema.")
        except Exception as e:
            pytest.fail(f"Schema Validation Failed: {e}")

    # --- STEP 3: ACCURACY CHECK (Golden Record) ---
    
    generated_text = result_dict.get("text", "")
    expected_text = golden_record.get("text", "")
    
    # EXACT MATCH check
    assert generated_text == expected_text, \
        f"Text Content Mismatch! Pipeline output does not match Golden Record exactly.\n" \
        f"Expected start: {repr(expected_text[:50])}\n" \
        f"Actual start:   {repr(generated_text[:50])}"
        
    print("   ✅ Step 3: Extracted text matches Golden Record EXACTLY.")

    # --- STEP 4: REAL EMBEDDING GENERATION ---
    print("   🧠 Step 4: Calling OpenAI API for Embeddings...")
    try:
        vector = embedder.generate_embedding(generated_text)
    except Exception as e:
        print(f"   ❌ API Error: {e}")
        pytest.fail(f"OpenAI API Call Failed. Check your Quota/Key.")
    
    # --- STEP 5: VERIFY VECTOR ---
    assert isinstance(vector, list), "Vector must be a list of floats"
    assert len(vector) == 1536, f"Vector dimension mismatch. Expected 1536 (text-embedding-3-small), got {len(vector)}"
    assert any(x != 0 for x in vector), "Vector contains real data (not empty)"
    
    print(f"   ✅ Step 5: Success! Received valid {len(vector)}-d vector from OpenAI.")