import pytest
import json
import re
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import YOUR class and YOUR models
# (We try to import QdrantDocument to use it for strict validation)
from pipeline import DataTransformer
try:
    from pipeline import QdrantDocument, MetadataModel
except ImportError:
    QdrantDocument = None
    print("⚠️ Warning: Could not import QdrantDocument/MetadataModel. Validation will be manual.")

# Load raw data
with open("data/raw_customer_api.json", "r") as f:
    RAW_DATA = json.load(f)

# Strict ISO 8601 Regex (YYYY-MM-DDTHH:MM:SSZ)
ISO_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"

@pytest.mark.filterwarnings("ignore::bs4.MarkupResemblesLocatorWarning")
def test_pipeline_output_satisfies_contract():
    """
    Contract Test: Validates that every successfully processed document
    strictly adheres to the QdrantDocument schema defined in your pipeline.
    """
    transformer = DataTransformer()
    print(f"\n🚀 Starting Contract Test on {len(RAW_DATA)} documents...")
    
    results = []
    for doc in RAW_DATA:
        # Your pipeline returns (result_dict, report_dict)
        processed, _ = transformer.process_document(doc)
        if processed:
            results.append(processed)
    
    assert len(results) > 0, "Pipeline returned no documents from valid input"

    for i, doc_dict in enumerate(results, 1):
        # --- 1. Schema Validation via Pydantic ---
        # If QdrantDocument is available, we use it to strictly validate the dictionary
        if QdrantDocument:
            try:
                validated_doc = QdrantDocument(**doc_dict)
                # Convert back to object for easy dot-notation access in assertions below
                payload = validated_doc
                meta = validated_doc.metadata
            except Exception as e:
                pytest.fail(f"Document {i} failed Pydantic validation: {e}")
        else:
            # Fallback for manual dict access if import failed
            payload = doc_dict # Dictionary
            meta = doc_dict.get("metadata", {})
        
        # --- 2. Logic & Field Name Validation ---
        
        # Handle difference between Object (dot) and Dict (bracket) access
        def get_val(obj, key):
            return getattr(obj, key) if not isinstance(obj, dict) else obj.get(key)

        doc_id = get_val(meta, "external_id")
        text = get_val(payload, "text")
        
        # Rule: Text must not be empty
        assert len(text.strip()) > 0, f"Text is empty for {doc_id}"
        
        # Rule: Mandatory Fields
        assert doc_id, "External ID cannot be empty"
        url = get_val(meta, "url")
        assert url and url.startswith("http"), f"URL must be absolute for {doc_id}"

        # Rule: Lists must be present (not None), even if empty
        tags = get_val(meta, "tags")
        sections = get_val(meta, "sections")
        categories = get_val(meta, "categories")
        
        assert isinstance(tags, list), f"Tags must be a list for {doc_id}"
        assert isinstance(sections, list), f"Sections must be a list for {doc_id}"
        assert isinstance(categories, list), f"Categories must be a list for {doc_id}"

        # Rule: Dates must be ISO 8601 UTC
        p_date = get_val(meta, "publish_date")
        if p_date:
            assert re.match(ISO_PATTERN, p_date), \
                f"Date {p_date} is not valid ISO 8601 for {doc_id}"

        print(f"✅ [{i}/{len(results)}] Schema Validated: {doc_id}")


def test_pipeline_edge_cases_and_resilience():
    """
    Resilience Test: Feeds specific 'broken' inputs to ensure the pipeline
    handles them gracefully (skips, defaults, or sanitizes).
    """
    transformer = DataTransformer()

    # 1. CASE: Missing Mandatory Field (External ID)
    doc_no_id = {
        "type": "story",
        "headlines": {"basic": "No ID"},
        "content_elements": [{"type": "text", "content": "Text"}],
        "canonical_website": "nj", # Required for URL logic
        "website_url": "/test"
    }
    res, report = transformer.process_document(doc_no_id)
    assert res is None, "Pipeline should SKIP document missing '_id'"
    assert report["reason"] == "Missing ID"

    # 2. CASE: Missing Mandatory Field (Text Content)
    doc_no_text = {
        "_id": "no_text_01",
        "canonical_website": "nj", # Required for URL logic
        "website_url": "/test",
        "content_elements": [] # Empty content
    }
    res, report = transformer.process_document(doc_no_text)
    assert res is None, "Pipeline should SKIP document with no text"
    assert report["reason"] == "Missing Text"

    # 3. CASE: Missing Optional Lists (Taxonomy)
    # Expected: Process success, but fields default to [] (not None)
    doc_no_tax = {
        "_id": "defaults_test",
        "canonical_website": "nj", # Required for URL logic
        "website_url": "/test",
        "content_elements": [{"type": "text", "content": "Valid text"}],
        # Completely missing 'taxonomy'
    }
    res, _ = transformer.process_document(doc_no_tax)
    assert res is not None
    
    # Check that defaults were applied (using Dict access since res is a dict)
    meta = res.get("metadata", {})
    assert meta.get("tags") == [], "Missing tags must default to []"
    assert meta.get("sections") == [], "Missing sections must default to []"
    assert meta.get("categories") == [], "Missing categories must default to []"

    # 4. CASE: Malformed Date
    # Expected: Either skip document OR sanitize date to None
    doc_bad_date = {
        "_id": "bad_date_01",
        "canonical_website": "nj", # Required for URL logic
        "website_url": "/test",
        "content_elements": [{"type": "text", "content": "Valid text"}],
        "publish_date": "This is garbage", 
    }
    res, _ = transformer.process_document(doc_bad_date)
    
    if res:
        meta = res.get("metadata", {})
        p_date = meta.get("publish_date")
        
        # If pipeline kept it, the date MUST be valid or None. It cannot be "garbage"
        if p_date:
            assert re.match(ISO_PATTERN, p_date), f"Pipeline allowed invalid date: {p_date}"
        else:
            assert p_date is None

    print("\n✅ Edge Case & Resilience tests passed!")