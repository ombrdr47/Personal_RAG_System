# src/memory.py
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import deque
import pickle

class ConversationMemory:
    """Manages conversation history and context"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.history = deque(maxlen=max_size)
        self.memory_file = "data/conversation_memory.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        
        # Load existing memory
        self.load_memory()
    
    def add_interaction(self, query: str, response: Dict[str, Any]):
        """Add a query-response pair to memory"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response['answer'],
            'query_type': response.get('query_type'),
            'sources': [
                {
                    'file_name': doc.metadata.get('file_name'),
                    'chunk_id': doc.metadata.get('chunk_id')
                }
                for doc in response.get('sources', [])
            ],
            'retrieval_quality': response.get('retrieval_quality')
        }
        
        self.history.append(interaction)
        self.save_memory()
    
    def get_context(self, n_recent: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        return list(self.history)[-n_recent:]
    
    def save_memory(self):
        """Persist memory to disk"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(list(self.history), f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def load_memory(self):
        """Load memory from disk"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.history = deque(data, maxlen=self.max_size)
        except Exception as e:
            print(f"Error loading memory: {e}")
    
    def clear(self):
        """Clear conversation memory"""
        self.history.clear()
        self.save_memory()

class UserPreferences:
    """Manages user preferences and learning"""
    
    def __init__(self):
        self.preferences_file = "data/user_preferences.json"
        self.preferences = {
            'preferred_sources': {},
            'query_patterns': {},
            'feedback_scores': {}
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.preferences_file), exist_ok=True)
        
        # Load existing preferences
        self.load_preferences()
    
    def update_source_preference(self, source: str, score: float):
        """Update preference for a source"""
        if source not in self.preferences['preferred_sources']:
            self.preferences['preferred_sources'][source] = []
        
        self.preferences['preferred_sources'][source].append(score)
        
        # Keep only recent scores
        if len(self.preferences['preferred_sources'][source]) > 10:
            self.preferences['preferred_sources'][source].pop(0)
        
        self.save_preferences()
    
    def get_source_score(self, source: str) -> float:
        """Get average preference score for a source"""
        scores = self.preferences['preferred_sources'].get(source, [])
        return sum(scores) / len(scores) if scores else 0.5
    
    def record_query_pattern(self, query_type: str):
        """Record query pattern"""
        if query_type not in self.preferences['query_patterns']:
            self.preferences['query_patterns'][query_type] = 0
        
        self.preferences['query_patterns'][query_type] += 1
        self.save_preferences()
    
    def save_preferences(self):
        """Save preferences to disk"""
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")
    
    def load_preferences(self):
        """Load preferences from disk"""
        try:
            if os.path.exists(self.preferences_file):
                with open(self.preferences_file, 'r') as f:
                    self.preferences = json.load(f)
        except Exception as e:
            print(f"Error loading preferences: {e}")

class KnowledgeGraph:
    """Simple knowledge graph for document relationships"""
    
    def __init__(self):
        self.graph_file = "data/knowledge_graph.pkl"
        self.graph = {
            'nodes': {},  # document/concept nodes
            'edges': []   # relationships
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.graph_file), exist_ok=True)
        
        # Load existing graph
        self.load_graph()
    
    def add_document_node(self, doc_id: str, metadata: Dict[str, Any]):
        """Add document node to graph"""
        self.graph['nodes'][doc_id] = {
            'type': 'document',
            'metadata': metadata,
            'connections': []
        }
    
    def add_concept_node(self, concept: str):
        """Add concept node to graph"""
        if concept not in self.graph['nodes']:
            self.graph['nodes'][concept] = {
                'type': 'concept',
                'connections': []
            }
    
    def add_edge(self, from_node: str, to_node: str, relationship: str):
        """Add relationship between nodes"""
        edge = {
            'from': from_node,
            'to': to_node,
            'relationship': relationship
        }
        
        self.graph['edges'].append(edge)
        
        # Update connections
        if from_node in self.graph['nodes']:
            self.graph['nodes'][from_node]['connections'].append(to_node)
        if to_node in self.graph['nodes']:
            self.graph['nodes'][to_node]['connections'].append(from_node)
        
        self.save_graph()
    
    def find_related_documents(self, doc_id: str, max_depth: int = 2) -> List[str]:
        """Find related documents through graph traversal"""
        if doc_id not in self.graph['nodes']:
            return []
        
        visited = set()
        to_visit = [(doc_id, 0)]
        related = []
        
        while to_visit:
            current, depth = to_visit.pop(0)
            
            if current in visited or depth > max_depth:
                continue
            
            visited.add(current)
            
            if current != doc_id and self.graph['nodes'][current]['type'] == 'document':
                related.append(current)
            
            # Add connections
            for conn in self.graph['nodes'][current]['connections']:
                if conn not in visited:
                    to_visit.append((conn, depth + 1))
        
        return related
    
    def save_graph(self):
        """Save graph to disk"""
        try:
            with open(self.graph_file, 'wb') as f:
                pickle.dump(self.graph, f)
        except Exception as e:
            print(f"Error saving graph: {e}")
    
    def load_graph(self):
        """Load graph from disk"""
        try:
            if os.path.exists(self.graph_file):
                with open(self.graph_file, 'rb') as f:
                    self.graph = pickle.load(f)
        except Exception as e:
            print(f"Error loading graph: {e}")

class PersistentMemoryManager:
    """Manages all memory components"""
    
    def __init__(self):
        # Set a default value for conversation memory size, or import from settings if available
        self.conversation_memory = ConversationMemory(
            max_size=10  # Replace 10 with your desired default size
        )
        self.user_preferences = UserPreferences()
        self.knowledge_graph = KnowledgeGraph()
    
    def record_interaction(self, query: str, response: Dict[str, Any]):
        """Record a complete interaction"""
        # Add to conversation memory
        self.conversation_memory.add_interaction(query, response)
        
        # Update user preferences
        self.user_preferences.record_query_pattern(response.get('query_type', 'unknown'))
        
        # Update source preferences based on quality
        for doc in response.get('sources', []):
            source = doc.metadata.get('file_name', 'unknown')
            score = response.get('retrieval_quality', 0.5)
            self.user_preferences.update_source_preference(source, score)
    
    def get_enhanced_context(self, query: str) -> Dict[str, Any]:
        """Get enhanced context for query"""
        return {
            'conversation_history': self.conversation_memory.get_context(),
            'user_preferences': self.user_preferences.preferences,
            'query_patterns': self.user_preferences.preferences.get('query_patterns', {})
        }
        # ✨ NEW METHOD
    def clear(self):
        """Clears all memory components."""
        print("Clearing all memory components...")
        self.conversation_memory.clear()
        
        # Clear user preferences and knowledge graph by deleting their files
        if os.path.exists(self.user_preferences.preferences_file):
            os.remove(self.user_preferences.preferences_file)
        self.user_preferences.__init__() # Re-initialize to default state
        
        if os.path.exists(self.knowledge_graph.graph_file):
            os.remove(self.knowledge_graph.graph_file)
        self.knowledge_graph.__init__() # Re-initialize to default state