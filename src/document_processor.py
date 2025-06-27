# src/document_processor.py

import os
import hashlib
from collections import Counter # ✨ FIX: Moved import to top
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader,
    JSONLoader, CSVLoader, UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import nltk
import spacy

# --- NLTK and SpaCy Setup ---
# This is fine here, as it only runs once on import.
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        
        # ✨ FIX: Use a more specific exception for loading the model.
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except (OSError, ImportError):
            print("Spacy 'en_core_web_sm' model not found. Please run 'python -m spacy download en_core_web_sm'")
            self.nlp = None
        
        self.loaders = {
            '.pdf': PyPDFLoader, '.docx': Docx2txtLoader, '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader, '.json': JSONLoader, '.csv': CSVLoader,
            '.html': UnstructuredHTMLLoader, '.htm': UnstructuredHTMLLoader
        }
    
    # --- Helper methods (no changes needed) ---
    def detect_file_type(self, file_path: str) -> str:
        return Path(file_path).suffix.lower()

    def extract_creation_date(self, file_path: str) -> str:
        stat = os.stat(file_path)
        return datetime.fromtimestamp(stat.st_ctime).isoformat()

    def detect_language(self, text: str) -> str:
        return "en"

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        if not text: return []
        words = [w for w in nltk.word_tokenize(text.lower()) if len(w) > 3 and w.isalnum()]
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(top_k)]

    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        entities = {"PERSON": [], "ORG": [], "LOC": [], "DATE": []}
        if self.nlp and text:
            doc = self.nlp(text[:1_000_000]) # Spacy has a character limit
            for ent in doc.ents:
                if ent.label_ in entities and ent.text.strip():
                    entities[ent.label_].append(ent.text.strip())
            for key in entities:
                entities[key] = list(set(entities[key]))
        return entities

    def analyze_structure(self, content: str) -> Dict[str, Any]:
        lines = content.split('\n')
        return {
            'total_lines': len(lines), 'total_words': len(content.split()),
            'total_chars': len(content),
            'has_headers': any(line.startswith('#') for line in lines),
            'has_lists': any(line.strip().startswith(('-', '*', '1.')) for line in lines)
        }

    def generate_doc_id(self, file_path: str, content: str) -> str:
        hash_input = f"{file_path}_{content[:100]}_{os.path.getsize(file_path)}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    # --- Core Parsing Logic ---
    def parse_document(self, file_path: str) -> List[Document]:
        file_type = self.detect_file_type(file_path)
        if file_type not in self.loaders:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        loader_class = self.loaders[file_type]
        if file_type == '.json':
            loader = loader_class(file_path, jq_schema='.', text_content=False)
        else:
            loader = loader_class(file_path)
        
        docs_from_loader = loader.load()
        if not docs_from_loader:
            return []
            
        full_content = "\n".join([doc.page_content for doc in docs_from_loader])
        
        # Create a base metadata dict for the entire file
        doc_metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_type': file_type,
            'file_size': os.path.getsize(file_path),
            'created_date': self.extract_creation_date(file_path),
            'doc_id': self.generate_doc_id(file_path, full_content),
            'language': self.detect_language(full_content),
            'keywords': self.extract_keywords(full_content),
            'entities': self.extract_named_entities(full_content),
            'structure': self.analyze_structure(full_content)
        }
        
        all_chunks = []
        for doc_idx, doc in enumerate(docs_from_loader):
            chunks = self.text_splitter.split_text(doc.page_content)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                # ✨ CRITICAL FIX: The robust way to create metadata
                # 1. Start with the loader's metadata (if any).
                # 2. Add our file-level metadata.
                # 3. Add/overwrite with our chunk-specific metadata to guarantee it exists.
                chunk_metadata = {
                    **(doc.metadata or {}),
                    **doc_metadata,
                    'chunk_id': f"{doc_metadata['doc_id']}_{doc_idx}_{chunk_idx}",
                    'chunk_index': chunk_idx,
                    'doc_index': doc_idx,
                    'chunk_size': len(chunk_text),
                    'total_chunks': len(chunks)
                }
                
                all_chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
        
        return all_chunks