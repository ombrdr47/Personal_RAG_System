# src/retrieval.py - Multiple solutions for CrossEncoder authentication

from typing import List, Dict, Any
from enum import Enum
import re
import numpy as np
import os

from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser

# ✨ SOLUTION 1: Try importing with fallback
try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except Exception as e:
    print(f"CrossEncoder import failed: {e}")
    CROSSENCODER_AVAILABLE = False
    # Fallback to sklearn
    from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from .vector_store import VectorStoreManager
from .memory import PersistentMemoryManager

class QueryType(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CODING = "coding"
    CONVERSATIONAL = "conversational"
    WEB_SEARCH = "web_search"

class QueryRouter:
    """Routes queries to appropriate retrieval strategies using LCEL."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.0
        )
        
        self.routing_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Classify the following query into one of these categories:
            - factual: Seeking specific facts or information likely contained in personal documents.
            - analytical: Requiring analysis or comparison of information likely in personal documents.
            - creative: Requiring creative generation.
            - coding: Related to programming or code.
            - conversational: General chat or discussion.
            - web_search: The query asks for real-time information, news, or general knowledge that is unlikely to be in personal documents (e.g., 'What is the weather today?', 'Who won the election?').
            
            Query: {query}
            
            Category:"""
        )
        
        self.routing_chain = self.routing_prompt | self.llm | StrOutputParser()
    
    def route(self, query: str) -> QueryType:
        """Determine query type for routing."""
        try:
            response = self.routing_chain.invoke({"query": query}).strip().lower()
            
            for query_type in QueryType:
                if query_type.value in response:
                    return query_type
            
            return QueryType.FACTUAL
        except Exception as e:
            print(f"Query routing failed: {e}")
            return QueryType.FACTUAL

class RobustCrossEncoder:
    """
    Robust CrossEncoder with multiple initialization strategies
    """
    
    def __init__(self, model_name='ms-marco-MiniLM-L-6-v2', embeddings=None):
        self.model_name = model_name
        self.embeddings = embeddings
        self.cross_encoder = None
        self.fallback_mode = False
        
        # ✨ SOLUTION 2: Try multiple initialization strategies
        self._initialize_cross_encoder()
    
    def _initialize_cross_encoder(self):
        """Try multiple strategies to initialize CrossEncoder"""
        
        if not CROSSENCODER_AVAILABLE:
            print("CrossEncoder not available, using embedding fallback")
            self.fallback_mode = True
            return
        
        strategies = [
            self._init_with_offline_mode,
            self._init_with_cache_dir,
            self._init_with_token,
            self._init_with_local_files_only
        ]
        
        for strategy in strategies:
            try:
                strategy()
                if self.cross_encoder is not None:
                    print(f"CrossEncoder initialized successfully with {strategy.__name__}")
                    return
            except Exception as e:
                print(f"{strategy.__name__} failed: {e}")
                continue
        
        print("All CrossEncoder initialization strategies failed, using fallback")
        self.fallback_mode = True
    
    def _init_with_offline_mode(self):
        """Try to initialize with offline mode"""
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        self.cross_encoder = CrossEncoder(self.model_name, device='cpu')
    
    def _init_with_cache_dir(self):
        """Try to initialize with custom cache directory"""
        cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        os.makedirs(cache_dir, exist_ok=True)
        self.cross_encoder = CrossEncoder(
            self.model_name, 
            device='cpu',
            cache_folder=cache_dir
        )
    
    def _init_with_token(self):
        """Try to initialize with HF token if available"""
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            self.cross_encoder = CrossEncoder(
                self.model_name, 
                device='cpu',
                use_auth_token=hf_token
            )
        else:
            raise ValueError("No HF token available")
    
    def _init_with_local_files_only(self):
        """Try to initialize with local_files_only=True"""
        self.cross_encoder = CrossEncoder(
            self.model_name, 
            device='cpu',
            local_files_only=True
        )
    
    def predict(self, pairs):
        """Predict scores for query-document pairs"""
        if self.fallback_mode or self.cross_encoder is None:
            return self._embedding_fallback(pairs)
        
        try:
            return self.cross_encoder.predict(pairs)
        except Exception as e:
            print(f"CrossEncoder prediction failed: {e}, falling back to embeddings")
            return self._embedding_fallback(pairs)
    
    def _embedding_fallback(self, pairs):
        """Fallback using embedding similarity"""
        if not self.embeddings:
            # Return neutral scores if no embeddings available
            return [0.5] * len(pairs)
        
        try:
            queries = [pair[0] for pair in pairs]
            documents = [pair[1] for pair in pairs]
            
            # Get embeddings
            query_embeddings = np.array(self.embeddings.embed_documents(queries))
            doc_embeddings = np.array(self.embeddings.embed_documents(documents))
            
            # Calculate cosine similarity for each pair
            similarities = []
            for i in range(len(pairs)):
                similarity = cosine_similarity(
                    query_embeddings[i:i+1], 
                    doc_embeddings[i:i+1]
                )[0][0]
                similarities.append(similarity)
            
            return similarities
        except Exception as e:
            print(f"Embedding fallback failed: {e}")
            return [0.5] * len(pairs)

class AdaptiveCRAG:
    """
    Implements an adaptive, corrective RAG pipeline with web search,
    re-ranking, and knowledge graph augmentation.
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager, embeddings: GoogleGenerativeAIEmbeddings, memory_manager: PersistentMemoryManager):
        self.vector_store = vector_store_manager
        self.embeddings = embeddings
        self.memory_manager = memory_manager
        self.query_router = QueryRouter()

        # ✨ SOLUTION 3: Use robust CrossEncoder with fallback
        # Try different models if one fails
        model_options = [
            'ms-marco-MiniLM-L-6-v2',
            'cross-encoder/ms-marco-MiniLM-L-12-v2',
            'cross-encoder/ms-marco-TinyBERT-L-2-v2'
        ]
        
        self.cross_encoder = None
        for model_name in model_options:
            try:
                self.cross_encoder = RobustCrossEncoder(
                    model_name=model_name,
                    embeddings=embeddings
                )
                if not self.cross_encoder.fallback_mode:
                    print(f"Successfully loaded CrossEncoder: {model_name}")
                    break
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue
        
        if self.cross_encoder is None or self.cross_encoder.fallback_mode:
            print("Using embedding fallback for re-ranking")
            self.cross_encoder = RobustCrossEncoder(embeddings=embeddings)
        
        self.web_search_tool = TavilySearchResults(k=settings.TOP_K)
        
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.7
        )
        
        self.generation_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""Based on the following context, provide a comprehensive and detailed answer to the query.
            If the context is empty or does not contain relevant information to answer the query, state that you could not find sufficient information.
            
            Context:
            {context}
            
            Query: {query}
            
            Answer:"""
        )
        
        self.generation_chain = self.generation_prompt | self.llm | StrOutputParser()

    def _augment_with_knowledge_graph(self, docs: List[Document]) -> List[Document]:
        """Augment retrieved docs with related docs from the knowledge graph."""
        if not docs:
            return []

        augmented_docs = {doc.metadata.get("chunk_id"): doc for doc in docs}
        top_doc_id = docs[0].metadata.get("doc_id")

        if top_doc_id:
            related_doc_ids = self.memory_manager.knowledge_graph.find_related_documents(top_doc_id)
            print(f"Found {len(related_doc_ids)} related doc IDs in KG: {related_doc_ids}")
        
        return list(augmented_docs.values())
    
    def generate_response(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate a response using an adaptive RAG pipeline.
        """
        route = self.query_router.route(query)
        
        documents = []
        quality_score = 0.0
        corrections_applied = False
        
        if route == QueryType.WEB_SEARCH:
            print("--- Routing to Web Search ---")
            web_results = self.web_search_tool.invoke(query)
            if web_results:
                documents = [
                    Document(page_content=res["content"], metadata={"file_name": "Web Search", "url": res["url"]})
                    for res in web_results
                ]
                quality_score = 1.0
        else:
            print(f"--- Routing to Local DB: {route.value} ---")
            initial_docs = self.vector_store.search(
                query,
                self.embeddings,
                k=settings.TOP_K * 4
            )

            if initial_docs:
                doc_pairs = [[query, doc.page_content] for doc in initial_docs]
                scores = self.cross_encoder.predict(doc_pairs)
                
                scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
                
                relevant_docs = [doc for score, doc in scored_docs if score >= settings.RELEVANCE_THRESHOLD]
                documents = relevant_docs[:settings.TOP_K]
                
                top_scores = [score for score, doc in scored_docs[:len(documents)]]
                quality_score = np.mean(top_scores) if top_scores else 0.0
                corrections_applied = True
                
                if documents:
                    documents = self._augment_with_knowledge_graph(documents)

        context = "\n\n".join(
            [f"[Source: {doc.metadata.get('file_name', 'Unknown')}]\n{doc.page_content}" for doc in documents]
        ) if documents else "No relevant information was found."
        
        response = self.generation_chain.invoke({"query": query, "context": context})
        
        return {
            'answer': response,
            'sources': documents,
            'query_type': route.value,
            'retrieval_quality': quality_score,
            'corrections_applied': corrections_applied
        }