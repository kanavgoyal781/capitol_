# #!/usr/bin/env python
# # coding: utf-8

# # In[ ]:


# import pytest
# import sys
# import os
# from hypothesis import given, settings, strategies as st

# # Add project root to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from pipeline import DataTransformer

# # --- 1. DATA GENERATION STRATEGIES ---
# # (These generate random structure to fuzz your pipeline)

# content_element_strategy = st.fixed_dictionaries({
#     "type": st.one_of(st.text(), st.none()),
#     "content": st.one_of(st.text(), st.none()),
#     "additional_properties": st.one_of(st.dictionaries(st.text(), st.text()), st.none())
# })

# taxonomy_strategy = st.fixed_dictionaries({
#     "tags": st.lists(
#         st.fixed_dictionaries({"slug": st.text(), "text": st.text()})
#     ),
#     "sections": st.lists(
#         st.fixed_dictionaries({"name": st.text(), "path": st.text()})
#     ),
#     "categories": st.lists(
#         st.fixed_dictionaries({"name": st.text(), "score": st.floats()})
#     )
# })

# raw_doc_strategy = st.fixed_dictionaries({
#     "_id": st.one_of(st.text(min_size=1), st.none()),
#     "type": st.text(),
#     "headlines": st.one_of(st.none(), st.fixed_dictionaries({"basic": st.text()})),
#     "content_elements": st.lists(content_element_strategy),
#     "taxonomy": st.one_of(st.none(), taxonomy_strategy),
#     "canonical_url": st.one_of(st.text(), st.none()),
#     "website_url": st.one_of(st.text(), st.none()),
#     "canonical_website": st.one_of(st.text(), st.none()),
#     "publish_date": st.one_of(st.text(), st.none()),
#     "first_publish_date": st.one_of(st.text(), st.none()),
#     "display_date": st.one_of(st.text(), st.none()),
#     "promo_items": st.one_of(st.none(), st.dictionaries(st.text(), st.text()))
# })

# # --- 2. THE TEST ---

# @settings(max_examples=100)
# @given(doc=raw_doc_strategy)
# def test_pipeline_resilience_to_garbage_data(doc):
#     """
#     Property Test: Fuzzes the pipeline with valid JSON structure but garbage content.
#     Passes if:
#     1. The pipeline DOES NOT CRASH (no unhandled exceptions).
#     2. Any result returned adheres to the basic flat schema.
#     """
#     transformer = DataTransformer()
    
#     try:
#         result, report = transformer.process_document(doc)
        
#         # If the pipeline successfully returns a document (didn't skip it),
#         # it MUST match your schema's structural rules.
#         if result:
#             assert isinstance(result, dict), "Result must be a dictionary"
            
#             # Direct Flat Access (No payload wrapper)
#             text = result.get("text")
#             metadata = result.get("metadata")
            
#             # Basic Schema Assertions
#             assert isinstance(text, str), "Output 'text' must be a string"
#             assert isinstance(metadata, dict), "Output 'metadata' must be a dict"
            
#             # Check a list field to ensure defaults (like []) are working
#             tags = metadata.get("tags")
#             assert isinstance(tags, list), "Metadata 'tags' must be a list (never None)"
            
#     except Exception as e:
#         # If we catch an exception here, the test fails.
#         # This proves the pipeline is not "crash-proof" against this specific input.
#         pytest.fail(f"Pipeline CRASHED on valid JSON structure with garbage content.\nError: {e}\nInput Doc: {doc}")


import pytest
import sys
import os
import json
from hypothesis import given, settings, strategies as st

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import DataTransformer

# --- 1. DATA GENERATION STRATEGIES ---
# (These remain the same, they successfully fuzz the input)

content_element_strategy = st.fixed_dictionaries({
    "type": st.one_of(st.text(), st.none()),
    "content": st.one_of(st.text(), st.none()),
    "additional_properties": st.one_of(st.dictionaries(st.text(), st.text()), st.none())
})

taxonomy_strategy = st.fixed_dictionaries({
    "tags": st.lists(
        st.fixed_dictionaries({"slug": st.text(), "text": st.text()})
    ),
    "sections": st.lists(
        st.fixed_dictionaries({"name": st.text(), "path": st.text()})
    ),
    "categories": st.lists(
        st.fixed_dictionaries({"name": st.text(), "score": st.floats()})
    )
})

raw_doc_strategy = st.fixed_dictionaries({
    # CRITICAL: Allow the _id field to be an empty string, which often causes issues in logging/extraction functions
    "_id": st.one_of(st.text(min_size=0), st.none()), 
    "type": st.text(),
    "headlines": st.one_of(st.none(), st.fixed_dictionaries({"basic": st.text()})),
    "content_elements": st.lists(content_element_strategy),
    "taxonomy": st.one_of(st.none(), taxonomy_strategy),
    "canonical_url": st.one_of(st.text(), st.none()),
    "website_url": st.one_of(st.text(), st.none()),
    "canonical_website": st.one_of(st.text(), st.none()),
    "publish_date": st.one_of(st.text(), st.none()),
    "first_publish_date": st.one_of(st.text(), st.none()),
    "display_date": st.one_of(st.text(), st.none()),
    "promo_items": st.one_of(st.none(), st.dictionaries(st.text(), st.text()))
})

# --- 2. THE TEST ---

@settings(max_examples=1000)
@given(doc=raw_doc_strategy)
def test_pipeline_resilience_to_garbage_data(doc):
    """
    Property Test: Fuzzes the pipeline for stability. If a crash occurs, 
    the test fails and prints the exact input document that caused it.
    """
    transformer = DataTransformer()
    
    try:
        # Use a print statement to capture the input right before the execution
        print(f"--- Attempting Input: {doc['_id'] if doc.get('_id') else 'NO_ID'} ---")

        result, report = transformer.process_document(doc)
        
        # If the pipeline successfully returns a document (didn't skip it),
        if result:
            assert isinstance(result, dict), "Result must be a dictionary"
            
            # Direct Flat Access (No payload wrapper)
            text = result.get("text")
            metadata = result.get("metadata")
            
            # Basic Schema Assertions
            assert isinstance(text, str), "Output 'text' must be a string"
            assert isinstance(metadata, dict), "Output 'metadata' must be a dict"
            
            # Check a list field to ensure defaults (like []) are working
            tags = metadata.get("tags")
            assert isinstance(tags, list), "Metadata 'tags' must be a list (never None)"
            
    except Exception as e:
        # --- CRASH HANDLER: This forces the crucial debug information to print ---
        print("\n\n######################################################################")
        print(f"🛑 CRASH DETECTED. The minimal failing input was:")
        print(json.dumps(doc, indent=2, ensure_ascii=False))
        print("######################################################################\n")
        
        # Fail the test with the error
        pytest.fail(f"Pipeline CRASHED on specific input. Error: {e}")