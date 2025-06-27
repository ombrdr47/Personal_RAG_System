import logging
import traceback
import aiofiles
import shutil
import asyncio  # ✨ FIX: Import asyncio for non-blocking calls
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse

from src.document_processor import DocumentProcessor
from src.embeddings import GeminiEmbedder
from src.vector_store import VectorStoreManager
from src.retrieval import AdaptiveCRAG
from src.memory import PersistentMemoryManager
from config.settings import settings

# --- Logger & App Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
app = FastAPI(title="Personal RAG System API", version="1.0.0")

@app.exception_handler(Exception)
async def unhandled(request, exc):
    logging.error(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Component Initialization ---
document_processor = DocumentProcessor(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
embedder = GeminiEmbedder()
vector_store = VectorStoreManager()
memory_manager = PersistentMemoryManager()
adaptive_crag = AdaptiveCRAG(
    vector_store_manager=vector_store,
    embeddings=embedder.get_embeddings_model(),
    memory_manager=memory_manager
)


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    use_memory: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query_type: str
    retrieval_quality: float
    corrections_applied: bool

class DocumentStats(BaseModel):
    total_documents: int
    collections: Dict[str, Any]
    last_updated: str # Keep this for consistency, even if not used everywhere

class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str


# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "Personal RAG System API", "status": "active"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        upload_dir = settings.DATA_DIR / "documents"
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / file.filename
        
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await file.read(1 << 20):
                await out.write(chunk)

        def _process_and_index(path: Path) -> int:
            docs = document_processor.parse_document(str(path))
            if not docs:
                return 0
            
            vector_store.add_documents(docs, embedder.get_embeddings_model())
            
            # ✨ FIX: Update KG efficiently, once per document, not per chunk.
            first_doc_meta = docs[0].metadata
            doc_id = first_doc_meta.get("doc_id")
            if doc_id:
                memory_manager.knowledge_graph.add_document_node(doc_id, first_doc_meta)
                for kw in first_doc_meta.get("keywords", []):
                    memory_manager.knowledge_graph.add_concept_node(kw)
                    memory_manager.knowledge_graph.add_edge(doc_id, kw, "contains_keyword")
            return len(docs)

        chunks_created = await run_in_threadpool(_process_and_index, dest)
        return UploadResponse(filename=file.filename, chunks_created=chunks_created, status="success")
    except Exception as e:
        # ✨ FIX: Removed duplicate raise statement
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {e}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # ✨ PERFORMANCE FIX: Run the blocking RAG chain in a separate thread.
        response_dict = await asyncio.to_thread(
            adaptive_crag.generate_response,
            query=request.query
            # The new generate_response doesn't use conversation_history directly,
            # but this is where you would pass it if you re-add that feature.
        )
        
        # ✨ FIX: Return the full metadata for more flexibility in the UI.
        sources = [
            {
                'file_name': doc.metadata.get('file_name', 'Unknown'),
                'content': doc.page_content, # Return full content
                'metadata': doc.metadata
            } for doc in response_dict.get('sources', [])
        ]
        
        return QueryResponse(
            answer=response_dict['answer'],
            sources=sources,
            query_type=response_dict['query_type'],
            retrieval_quality=response_dict['retrieval_quality'],
            corrections_applied=response_dict['corrections_applied']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}")

@app.get("/stats", response_model=DocumentStats)
async def get_stats():
    try:
        collection_stats = vector_store.get_collection_stats()
        unified_collection_name = vector_store.collection_names['all']
        total_docs = collection_stats.get(unified_collection_name, {}).get('count', 0)
        return DocumentStats(total_documents=total_docs, collections=collection_stats, last_updated=datetime.now().isoformat())
    except Exception as e:
        # This will catch the error if the client handle is invalid
        logging.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")
    

@app.delete("/clear")
async def clear_database():
    """
    Clears all data by resetting the DB and re-initializing stateful components.
    """
    # ✨ FIX: This is the final, correct implementation for clearing state.
    global vector_store, memory_manager, adaptive_crag
    
    try:
        print("--- Clearing all data and re-initializing state ---")
        
        # 1. Wipe the database on disk.
        vector_store.client.reset()
        
        # 2. Delete all other persisted files (memory, docs, etc.)
        if settings.DATA_DIR.exists():
            shutil.rmtree(settings.DATA_DIR)
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # 3. Re-create fresh instances of all stateful managers.
        #    This gives them new, valid client handles and empty states.
        print("Re-creating stateful managers...")
        new_vector_store = VectorStoreManager()
        new_memory_manager = PersistentMemoryManager()
        
        # 4. Re-assign the application-wide global variables to the new instances.
        vector_store = new_vector_store
        memory_manager = new_memory_manager
        
        # 5. MOST IMPORTANT: Re-create the CRAG object so it gets the
        #    reference to the NEW vector_store and memory_manager.
        print("Re-creating AdaptiveCRAG with new managers...")
        adaptive_crag = AdaptiveCRAG(
            vector_store_manager=new_vector_store,
            embeddings=embedder.get_embeddings_model(),
            memory_manager=new_memory_manager
        )
        
        print("--- State re-initialization complete ---")
        return {"status": "success", "message": "All data and components re-initialized."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {e}")
    
    
# --- Memory Endpoints ---
@app.get("/memory/conversation")
async def get_conversation_history():
    history = memory_manager.conversation_memory.get_context(n_recent=20)
    return {"history": history}

@app.get("/memory/preferences")
async def get_user_preferences():
    preferences = memory_manager.user_preferences.preferences
    return {"preferences": preferences}


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app" if __name__ == "__main__" else __name__, host=settings.API_HOST, port=settings.API_PORT, reload=True)