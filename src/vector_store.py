# src/vector_store.py

import os
import uuid
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import settings

class VectorStoreManager:
    def __init__(self, persist_directory: str = None):
        self.persist_directory = str(persist_directory or settings.CHROMA_PERSIST_DIR)
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True)
        )

        self.collection_names = {
            'documents': 'doc_collection', 'code': 'code_collection',
            'notes': 'notes_collection', 'web': 'web_collection',
            'all': 'unified_collection'
        }
        
        self.vector_stores = {}
        self._initialize_collections()

    def _initialize_collections(self):
        """Ensures all defined collections exist in the database at startup."""
        print("Initializing and verifying vector store collections...")
        for collection_name in self.collection_names.values():
            self.client.get_or_create_collection(name=collection_name)
        print("All collections are ready.")

    # ✨ FINAL FIX (Part 1): Restore the _clean_metadata helper function.
    @staticmethod
    def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Converts all metadata values to a ChromaDB-compatible scalar type."""
        cleaned_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                cleaned_metadata[key] = value
            else:
                # Convert lists, dicts, and other types to a string representation.
                cleaned_metadata[key] = str(value)
        return cleaned_metadata

    def add_documents(self, documents: List[Document], embeddings: GoogleGenerativeAIEmbeddings):
        """Categorizes docs and adds them with cleaned metadata and explicit IDs."""
        if not documents: return

        categorized_docs = {name: [] for name in self.collection_names.values()}
        for doc in documents:
            category_key = self.categorize_document(doc)
            collection_name = self.collection_names.get(category_key, 'documents')
            categorized_docs[collection_name].append(doc)
        
        categorized_docs['unified_collection'] = documents

        for collection_name, docs_list in categorized_docs.items():
            if not docs_list: continue
            
            vector_store = self.get_or_create_collection(collection_name, embeddings)
            texts = [doc.page_content for doc in docs_list]
            
            # ✨ FINAL FIX (Part 2): Use the _clean_metadata function here.
            metadatas = [self._clean_metadata(doc.metadata) for doc in docs_list]
            
            ids = [meta.get("chunk_id") or str(uuid.uuid4()) for meta in metadatas]

            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def get_or_create_collection(self, collection_name: str, embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
        if collection_name not in self.vector_stores:
            self.vector_stores[collection_name] = Chroma(
                client=self.client, collection_name=collection_name,
                embedding_function=embeddings, persist_directory=self.persist_directory
            )
        return self.vector_stores[collection_name]

    def categorize_document(self, document: Document) -> str:
        file_type = document.metadata.get('file_type', '').lower()
        file_path = document.metadata.get('file_path', '').lower()
        if file_type in ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.ts', '.jsx']:
            return 'code'
        if 'notes' in file_path and file_type in ['.md', '.txt']:
            return 'notes'
        if file_type in ['.html', '.htm'] or 'url' in document.metadata:
            return 'web'
        return 'documents'

    def search(self, query: str, embeddings: GoogleGenerativeAIEmbeddings, k: int = 5, collection_name: str = None) -> List[Document]:
        target_collection = collection_name or self.collection_names['all']
        vector_store = self.get_or_create_collection(target_collection, embeddings)
        return vector_store.similarity_search(query, k=k)
    
    def fetch_documents_by_id(self, ids: List[str], collection_name: str = None) -> List[Document]:
        target_collection = collection_name or self.collection_names['all']
        try:
            collection = self.client.get_collection(target_collection)
            results = collection.get(ids=ids, include=["metadatas", "documents"])
            if not results or not results.get('ids'): return []
            return [Document(page_content=c, metadata=m) for c, m in zip(results['documents'], results['metadatas'])]
        except Exception as e:
            print(f"Error fetching documents by ID from '{target_collection}': {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        stats = {}
        for pretty_name, collection_id in self.collection_names.items():
            try:
                collection = self.client.get_collection(collection_id)
                stats[collection_id] = {'count': collection.count()}
            except Exception:
                stats[collection_id] = {'count': 0}
        return stats