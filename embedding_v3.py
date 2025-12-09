#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import logging
from typing import List
from openai import OpenAI

logger = logging.getLogger("CapitolPipeline")

class EmbeddingModel:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-small"

    def generate_embedding(self, text: str) -> List[float]:
        # This is the "granular" function app.py needs
        if not text: return []
        try:
            res = self.client.embeddings.create(model=self.model, input=text)
            return res.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return []

