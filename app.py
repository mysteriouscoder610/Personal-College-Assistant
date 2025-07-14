import streamlit as st
from summarizer_module import show_summarizer_page
from rag_module import show_rag_page

# Page configuration
st.set_page_config(
    page_title="Personal College Assistant",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents' not in st.session_state:
    st.session_state.documents = None

# Main app
def main():
    st.title("🎓 Personal College Assistant")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["Document Summarizer", "RAG Query System"]
    )
    
    # Check for API key
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        if not google_api_key or google_api_key == "your_gemini_api_key_here":
            st.error("⚠️ Please set your GOOGLE_API_KEY in Streamlit secrets")
            st.stop()
    except KeyError:
        st.error("⚠️ GOOGLE_API_KEY not found in Streamlit secrets")
        st.stop()
    
    if page == "Document Summarizer":
        show_summarizer_page()
    
    elif page == "RAG Query System":
        show_rag_page()

if __name__ == "__main__":
    main() 