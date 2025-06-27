#src/embeddings.py

import time
from typing import List, Optional
import numpy as np
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings.base import Embeddings
from config.settings import settings

class GeminiEmbedder:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.GOOGLE_API_KEY
        genai.configure(api_key=self.api_key)
        
        # Use LangChain's Google Generative AI Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=self.api_key
        )
        
        # Rate limiting parameters
        self.requests_per_minute = 60
        self.last_request_time = 0
        self.min_request_interval = 60.0 / self.requests_per_minute
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        self._rate_limit()
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        embeddings = []
        
        # Process in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            self._rate_limit()
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def get_embeddings_model(self) -> Embeddings:
        """Return the LangChain embeddings model"""
        return self.embeddings