#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[28]:


import json
import logging
import csv
import re
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError

class MetadataModel(BaseModel):
    # optional fields
    title: str | None = None
    url: str
    external_id: str
    publish_date: str | None = None
    datetime: str | None = None
    first_publish_date: str | None = None
    website: str | None = None
    sections: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    thumb: str | None = None

    # list fields


class QdrantDocument(BaseModel):
    text: str
    metadata: MetadataModel

import json
import logging
import csv
import re
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# --- 1. SETUP LOGGING ---
# This logger captures the "story" of each document
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log", mode='w', encoding='utf-8'), # 'w' overwrites log each run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CapitolPipeline")

class DataTransformer:
    def __init__(self):
        # We can track state here if needed (e.g. total processed count)
        pass

    # --- YOUR EXTRACTION LOGIC (Preserved Exactly) ---
    
    # def clean_text(self, raw_string):
    #     if not raw_string:
    #         return ""
    #     # Adding separator=" " avoids words merging (e.g. "end.Start")
    #     return BeautifulSoup(raw_string, "html.parser").get_text(separator=" ").strip()

    # def clean_html(self, raw_html):
    #     """Helper: Removes HTML tags and returns clean text."""
    #     if not raw_html:
    #         return ""
    #     soup = BeautifulSoup(raw_html, "html.parser")
    #     return soup.get_text(separator=" ").strip()
    def clean_text(self, raw_string):
        if raw_string is None:
            return ""
        
        # Defensive Coding
        if not isinstance(raw_string, str):
            raw_string = str(raw_string)
    
        # CHANGE HERE: Use \n separator to preserve paragraph structure
        # and strip=True to remove leading/trailing whitespace from each block
        soup = BeautifulSoup(raw_string, "html.parser")
        text = soup.get_text(separator="\n")
        
        return text

    def clean_html_old(self, raw_html):
        """Helper: Removes HTML tags and returns clean text."""
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        
        # ❌ OLD BROKEN CODE:
        # text = soup.get_text(separator=" ")
        # return " ".join(text.split())
    
        # ✅ NEW CORRECT CODE (Matches clean_text):
        text = soup.get_text(separator="\n")
        return text

    def clean_html(self, raw_html):
        """Helper: Removes HTML tags and returns clean text."""
        if not raw_html:
            return "", False
        soup = BeautifulSoup(raw_html, "html.parser")

        tags_found = [tag.name for tag in soup.find_all()]
        # if tags_found:
        #     print(f"DEBUG: Found HTML tags: {tags_found}")
        # else:
        #     print("DEBUG: No HTML tags found (Plain text).")
    
        text = soup.get_text(separator="\n", strip=True)
        return text, tags_found

    def get_text_body_old_working(self, raw_doc):
        """
        Loops through 'content_elements' to build the full article text.
        """
        text_parts = []
        elements = raw_doc.get('content_elements', [])
        
        for element in elements:
            el_type = element.get('type', '')
            
            # Case A: Standard Text or Headers
            if el_type in ['text']:
                content = element.get('content', '')
                if content:
                    text_parts.append(self.clean_html(content))
        
        # Join all parts with double newlines to separate paragraphs
        return " ".join(text_parts)

    def get_text_body_aj_test(self, raw_doc):
        """
        Loops through 'content_elements' to build the full article text.
        """
        text_parts = []
        elements = raw_doc.get('content_elements', [])

        full_text = ""
        is_prev_html = False
        
        for element in elements:
            el_type = element.get('type', '')
            
            # Case A: Standard Text or Headers
            if el_type in ['text']:
                content = element.get('content', '')
                if content:
                    # text_parts.append(self.clean_html(content))
                    cleaned_text, is_html = self.clean_html(content)
                    if is_html:
                        full_text = full_text + "\n" + cleaned_text.strip()
                        is_prev_html = True
                    elif is_prev_html:
                        full_text = full_text + "\n" + cleaned_text.strip()
                        is_prev_html = False
                    else:
                        full_text = full_text + " " + cleaned_text.strip()
        
        # Join all parts with double newlines to separate paragraphs
        # return " ".join(text_parts)
        return full_text

    def get_text_body(self, raw_doc):
        """
        Loops through 'content_elements' to build the full article text.
        """
        text_parts = []
        elements = raw_doc.get('content_elements', [])

        full_text = ""
        full_text_with_html = ""
        is_prev_html = False
        
        for element in elements:
            el_type = element.get('type', '')
            
            # Case A: Standard Text or Headers
            if el_type in ['text']:
                content = element.get('content', '')
                if content:
                    full_text_with_html = full_text_with_html + " " + content
                    full_text, is_html = self.clean_html(full_text_with_html)
        
        return full_text.strip()

    def extract_title(self, article):
        # Navigate to headlines -> basic
        raw_title = article.get('headlines', {}).get('basic', '')
        return self.clean_text(raw_title)

    def extract_url_general(self, raw_doc):
        """
        Dynamically builds the URL for ANY website found in the data.
        """
        # 1. Get the path (e.g., "/news/article123")
        relative_path = raw_doc.get('website_url', '')
        
        # 2. Get the site identifier directly from the data
        site_id = raw_doc.get('canonical_website')
        
        # Safety check: If site_id is missing, return the relative path as-is
        if not site_id:
            return relative_path
            
        # 3. Construct the domain dynamically
        base_domain = f"https://www.{site_id}.com"
        
        # 4. Combine them if it's a relative path
        if relative_path.startswith('/'):
            return base_domain + relative_path
        
        return relative_path

    def extract_id(self, raw_doc):
        """
        Extracts the unique identifier (_id) from the raw document.
        """
        doc_id = raw_doc.get('_id')
        if doc_id is None:
            return None
        return str(doc_id)

    def format_date_iso(self, date_str: Any) -> Optional[str]:
            """
            Normalize to ISO 8601 UTC.
            """
            if not date_str or not isinstance(date_str, str) or date_str[-1] != "Z":
                return None
    
            try:
                return date_str
    
            except (ValueError, TypeError):
                return None

    def extract_publish_date(self, raw_doc):
        # Try root level first, then additional_properties
        date_str = raw_doc.get('publish_date')
        return self.format_date_iso(date_str)

    def extract_first_publish_date(self, raw_doc):
        date_str = raw_doc.get('first_publish_date')
        return self.format_date_iso(date_str)

    def extract_datetime(self, raw_doc):
        """
        Extracts the main datetime.
        Schema: 'Typically the same as publish_date'
        """
        return self.extract_publish_date(raw_doc)

    def extract_website(self, raw_doc):
        # 'canonical_website' is the authoritative field in the raw data
        site = raw_doc.get('canonical_website')
        return str(site) if site else None

    def extract_sections(self, raw_doc):
        """
        Extracts the list of section names from the document's taxonomy.
        """
        taxonomy = raw_doc.get('taxonomy', {})
        if not taxonomy:
            return []

        raw_sections = taxonomy.get('sections', [])
        
        if not isinstance(raw_sections, list):
            return []
        
        section_names = []
        for section in raw_sections:
            if isinstance(section, dict):
                name = section.get('name')
                if name and isinstance(name, str):
                    section_names.append(name.strip())
                    
        return list(dict.fromkeys(section_names))

    def extract_categories(self, raw_doc):
        taxonomy = raw_doc.get('taxonomy', {})
        if not taxonomy:
            return []

        raw_categories = taxonomy.get('categories', [])
        if not isinstance(raw_categories, list):
            return []

        iab_cats = [
            c for c in raw_categories
            if isinstance(c, dict) and c.get('classifier') == 'iab_content_taxonomy'
        ]

        iab_cats.sort(key=lambda x: x.get('score', 0) or 0, reverse=True)

        final_categories = []
        seen = set()

        for cat in iab_cats:
            name = cat.get('name')
            if not name:
                continue

            cleaned = self.clean_text(name)
            if cleaned in seen:
                continue 

            seen.add(cleaned)
            final_categories.append(cleaned)

            if len(final_categories) == 5:
                break

        return final_categories

    def extract_tags(self, raw_doc):
        """
        Extracts tags for metadata.tags.
        """
        taxonomy = raw_doc.get("taxonomy", {})
        if not isinstance(taxonomy, dict):
            return []

        raw_tags = taxonomy.get("tags", [])
        if not isinstance(raw_tags, list):
            return []

        tags = []
        seen = set()

        for tag in raw_tags:
            if not isinstance(tag, dict):
                continue

            slug = tag.get("slug")
            if not isinstance(slug, str):
                continue

            slug = slug.strip()
            if not slug or slug in seen:
                continue

            seen.add(slug)
            tags.append(slug)

            if len(tags) == 5:
                break

        return tags

    def extract_thumb(self, raw_doc):
        """
        Extracts metadata.thumb (optional).
        Converts relative resizeUrl to absolute URL.
        """
        promo_items = raw_doc.get("promo_items") or {}
        if not isinstance(promo_items, dict): return None

        basic = promo_items.get("basic")
        if not isinstance(basic, dict): return None

        add_props = basic.get("additional_properties") or {}
        if not isinstance(add_props, dict): return None

        rel = add_props.get("resizeUrl")
        site = raw_doc.get("canonical_website")

        if isinstance(rel, str) and rel.strip() and isinstance(site, str) and site.strip():
            return f"https://www.{site.strip().lower()}.com{rel.strip()}"

        return None

    # --- MAIN PROCESSOR WITH DETAILED LOGGING ---
    # --- MAIN PROCESSOR WITH DETAILED LOGGING ---
    def process_document(self, raw_doc: dict) -> tuple[Optional[Dict], Dict]:
        """
        Orchestrates transformation with comprehensive logging for ALL fields.
        Returns: (Processed Document OR None, Report Dictionary)
        """
        doc_id = raw_doc.get('_id', 'UNKNOWN_ID')
        report: Dict[str, Any] = {"id": doc_id, "status": "SKIPPED", "reason": "Unknown"}

        logger.info(f"📄 Processing Document ID: {doc_id}")

        # --- 1. CRITICAL CHECKS (Fail if missing) ---
        external_id = self.extract_id(raw_doc)
        if not external_id:
            logger.error("❌ SKIPPING: Document missing '_id'.")
            report["reason"] = "Missing ID"
            return None, report

        url = self.extract_url_general(raw_doc)
        if not url:
            logger.warning(f"   ⚠️ SKIPPING {doc_id}: Required field 'url' not found.")
            report["reason"] = "Missing URL"
            return None, report
        logger.info(f"   ✅ URL found: {url}")

        text_content = self.get_text_body(raw_doc)
        if not text_content or not text_content.strip():
            logger.warning(f"   ⚠️ SKIPPING {doc_id}: Required field 'text' is empty.")
            report["reason"] = "Missing Text"
            return None, report
        logger.info(f"   ✅ Text body found ({len(text_content)} chars).")

        # --- 2. OPTIONAL FIELD CHECKS (Log existence) ---
        title = self.extract_title(raw_doc)
        if title:
            logger.info(f"   ☑️ Title found: '{title[:30]}...'")
        else:
            logger.info("   ⚪ Title NOT found.")

        pub_date = self.extract_publish_date(raw_doc)
        if pub_date:
            logger.info(f"   ☑️ Publish Date found: {pub_date}")
        else:
            logger.info("   ⚪ Publish Date NOT found.")

        first_pub = self.extract_first_publish_date(raw_doc)
        if first_pub:
            logger.info(f"   ☑️ First Publish Date found: {first_pub}")
        else:
            logger.info("   ⚪ First Publish Date NOT found.")

        doc_time = self.extract_datetime(raw_doc)
        if doc_time:
            logger.info(f"   ☑️ Datetime found: {doc_time}")
        else:
            logger.info("   ⚪ Datetime NOT found.")

        website = self.extract_website(raw_doc)
        if website:
            logger.info(f"   ☑️ Website found: {website}")
        else:
            logger.info("   ⚪ Website NOT found.")

        thumb = self.extract_thumb(raw_doc)
        if thumb:
            logger.info("   ☑️ Thumbnail found.")
        else:
            logger.info("   ⚪ Thumbnail NOT found.")

        # Taxonomy lists
        sections = self.extract_sections(raw_doc)
        categories = self.extract_categories(raw_doc)
        tags = self.extract_tags(raw_doc)

        if sections:
            logger.info(f"   ☑️ Sections found: {len(sections)} items")
        else:
            logger.info("   ⚪ Sections empty.")

        if categories:
            logger.info(f"   ☑️ Categories found: {len(categories)} items")
        else:
            logger.info("   ⚪ Categories empty.")

        if tags:
            logger.info(f"   ☑️ Tags found: {len(tags)} items")
        else:
            logger.info("   ⚪ Tags empty.")

        # --- 3. BUILD METADATA ---
        try:
            # Build in the exact order you want to see in JSON
            metadata: Dict[str, Any] = {}

            # 1. Core identity
            if title:
                metadata["title"] = title
            metadata["url"] = url
            metadata["external_id"] = external_id

            # 2. Dates
            if pub_date:
                metadata["publish_date"] = pub_date
            if doc_time:
                metadata["datetime"] = doc_time
            if first_pub:
                metadata["first_publish_date"] = first_pub

            # 3. Source
            if website:
                metadata["website"] = website

            # 4. Taxonomy lists (always included, even if empty)
            metadata["sections"] = sections
            metadata["categories"] = categories
            metadata["tags"] = tags

            # 5. Media
            if thumb:
                metadata["thumb"] = thumb

            try:
                meta_model = MetadataModel(**metadata)
                doc_model = QdrantDocument(
                    text=text_content,
                    metadata=meta_model,
                )
            except ValidationError as ve:
                logger.error(f"   ❌ Validation failed for {doc_id}: {ve}")
                report["reason"] = "Schema validation failed"
                return None, report

            logger.info(f"   🎉 Document {doc_id} successfully transformed.")
            report["status"] = "SUCCESS"
            report["reason"] = ""

            # exclude_none=True ensures we don't output "title": null
            return doc_model.model_dump(exclude_none=True), report

        except Exception as e:
            logger.error(f"   💥 CRITICAL ERROR on {doc_id}: {e}")
            report["reason"] = f"Crash: {str(e)}"
            return None, report
dead_letter_path = "dead_letter_queue.jsonl"
# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        with open("raw_customer_api.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: 'raw_customer_api.json' not found.")
        exit(1)

    transformer = DataTransformer()
    valid_docs = []
    report_data = []
    
    # [CHANGE 3: Added seen_ids set for duplicate handling]
    seen_ids = set()

    print(f"🚀 Starting ingestion of {len(raw_data)} documents...")

    for doc in raw_data:
        # Check for duplicates BEFORE processing
        doc_id = doc.get('_id')
        if doc_id in seen_ids:
            logger.warning(f"⚠️ SKIPPING {doc_id}: Duplicate external_id found.")
            report_data.append({
                "id": doc_id,
                "status": "SKIPPED",
                "reason": "Duplicate external_id"
            })
            continue
        
        if doc_id:
            seen_ids.add(doc_id)

        # Process document
        processed_doc, report = transformer.process_document(doc)
        report_data.append(report)
        if processed_doc:
            valid_docs.append(processed_doc)
        else:
        # anything that failed validation / critical checks
            with open(dead_letter_path, "a", encoding="utf-8") as dl:
                record = {
                "id": doc_id,
                "reason": report.get("reason", "unknown"),
                "raw_doc": doc,
                }
                dl.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Save Valid Output (JSON)
    with open("processed_output_updated_2.json", "w", encoding="utf-8") as f:
        json.dump(valid_docs, f, indent=2, ensure_ascii=False)

    # Save Summary Report (CSV)
    with open("ingestion_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "status", "reason"])
        writer.writeheader()
        writer.writerows(report_data)

    print("\n✅ Processing Complete!")
    print(f"   - Valid Docs: {len(valid_docs)} (Saved to processed_output_updated.json)")
    print(f"   - Failed/Skipped: {len(raw_data) - len(valid_docs)}")
    print(f"   - Detailed Logs: pipeline.log")
    print(f"   - Summary Report: ingestion_report.csv")


# In[29]:


def execute_transformation_step(input_file: str, output_file: str) -> List[Dict]:
    transformer = DataTransformer()
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    
    valid_docs = []
    seen_ids = set()
    
    for doc in raw_data:
        doc_id = doc.get('_id')
        if doc_id in seen_ids: 
            continue
        if doc_id: 
            seen_ids.add(doc_id)
        
        # FIX: Unpack the tuple (result, report)
        result, report = transformer.process_document(doc)
        
        # FIX: Check result and append directly (it is already a dict, not a model)
        if result: 
            valid_docs.append(result)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(valid_docs, f, indent=2, ensure_ascii=False)
    
    return valid_docs


# In[ ]:




