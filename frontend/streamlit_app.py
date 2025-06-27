# streamlit_app.py

import streamlit as st
import requests
import json
from typing import List, Dict, Any
import os

# API Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Personal RAG System",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Title and description
st.title("🧠 Personal RAG System")
st.markdown("*Advanced Retrieval-Augmented Generation with Corrective RAG and Adaptive Routing*")

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pdf', 'docx', 'txt', 'md', 'json', 'csv', 'html'],
        help="Supported formats: PDF, DOCX, TXT, MD, JSON, CSV, HTML"
    )
    
    if uploaded_file is not None:
        with st.spinner('Processing document...'):
            files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_URL}/upload", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Uploaded {result['filename']}")
                st.info(f"Created {result['chunks_created']} chunks")
            else:
                st.error("Failed to upload document")
    
    st.divider()
    
    # System stats
    st.header("📊 System Statistics")
    
    try:
        stats_response = requests.get(f"{API_URL}/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            
            st.metric("Total Documents", stats['total_documents'])
            
            st.subheader("Collections")
            for name, info in stats['collections'].items():
                st.write(f"**{name}**: {info['count']} chunks")
    except:
        st.error("Failed to fetch statistics")
    
    st.divider()
    
    # Clear database
    if st.button("🗑️ Clear Database", type="secondary"):
        # Using a checkbox for confirmation is good practice
        if st.checkbox("Are you sure you want to delete everything?"):
            response = requests.delete(f"{API_URL}/clear")
            if response.status_code == 200:
                st.success("✅ Database cleared successfully!")
                
                # ✨ FIX: Clear all of Streamlit's caches
                st.cache_data.clear()
                st.cache_resource.clear()
                
                # Rerun the app to reflect the changes immediately
                st.rerun()
            else:
                st.error("Failed to clear database.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Query Interface")
    
    # Query input
    query = st.text_area(
        "Enter your query:",
        placeholder="Ask anything about your documents...",
        height=100
    )
    
    # Query options
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        use_memory = st.checkbox("Use conversation memory", value=True)
    with col_opt2:
        k_value = st.slider("Number of sources", 1, 10, 5)
    
    # Submit button
    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner('Searching and generating response...'):
                # Make API request
                request_data = {
                    "query": query,
                    "k": k_value,
                    "use_memory": use_memory
                }
                
                response = requests.post(
                    f"{API_URL}/query",
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'query': query,
                        'response': result
                    })
                    
                    # Display response
                    st.markdown("### 📝 Response")
                    st.write(result['answer'])
                    
                    # Display metrics
                    st.markdown("### 📈 Retrieval Metrics")
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Query Type", result['query_type'].title())
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with metric_cols[1]:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Retrieval Quality", f"{result['retrieval_quality']:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with metric_cols[2]:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Corrections Applied", "Yes" if result['corrections_applied'] else "No")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display sources
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(result['sources']):
                        with st.expander(f"Source {i+1}: {source['file_name']}"):
                            st.write(source['content'])
                            st.json(source['metadata'])
                else:
                    st.error("Failed to get response")
        else:
            st.warning("Please enter a query")

with col2:
    st.header("📜 Conversation History")
    
    if st.session_state.conversation_history:
        for i, item in enumerate(reversed(st.session_state.conversation_history[-5:])):
            with st.expander(f"Q: {item['query'][:50]}..."):
                st.write("**Query:**", item['query'])
                st.write("**Answer:**", item['response']['answer'][:200] + "...")
                st.write("**Type:**", item['response']['query_type'])
                st.write("**Quality:**", f"{item['response']['retrieval_quality']:.2f}")
    else:
        st.info("No conversation history yet")
    
    # Memory insights
    st.header("🧠 Memory Insights")
    
    try:
        # Get conversation memory
        conv_response = requests.get(f"{API_URL}/memory/conversation")
        if conv_response.status_code == 200:
            conv_history = conv_response.json()['history']
            st.metric("Stored Conversations", len(conv_history))
        
        # Get user preferences
        pref_response = requests.get(f"{API_URL}/memory/preferences")
        if pref_response.status_code == 200:
            preferences = pref_response.json()['preferences']
            
            # Display query patterns
            if preferences.get('query_patterns'):
                st.subheader("Query Patterns")
                for pattern, count in preferences['query_patterns'].items():
                    st.write(f"**{pattern}**: {count} queries")
    except:
        st.info("Memory data not available")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Personal RAG System v1.0 | Powered by Google Gemini & LangChain</p>
</div>
""", unsafe_allow_html=True)