import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
import tempfile
import os
import asyncio
import threading
from functools import wraps

# Fix for asyncio event loop issues in Streamlit
def fix_asyncio_loop():
    """Fix asyncio event loop issues in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Apply the fix at module level
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    st.warning("nest_asyncio not installed. Some features may not work properly.")
    pass

def load_file_for_rag(uploaded_file):
    """Load document from uploaded file for RAG."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(tmp_file_path)
        elif file_extension == 'txt':
            loader = TextLoader(tmp_file_path, encoding='utf-8')
        elif file_extension in ['docx', 'doc']:
            loader = Docx2txtLoader(tmp_file_path)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
        
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return documents
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_vector_store(documents):
    """Create FAISS vector store from documents."""
    try:
        # Fix asyncio event loop before creating embeddings
        fix_asyncio_loop()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Get API key with better error handling
        google_api_key = None
        
        # Try to get from session state first
        if hasattr(st.session_state, 'google_api_key') and st.session_state.google_api_key:
            google_api_key = st.session_state.google_api_key
        # Fallback to environment variable
        elif os.getenv("GOOGLE_API_KEY"):
            google_api_key = os.getenv("GOOGLE_API_KEY")
        # Try Streamlit secrets
        elif hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
            google_api_key = st.secrets["GOOGLE_API_KEY"]
        
        if not google_api_key:
            st.error("❌ Google API Key not found. Please set it in your environment variables or Streamlit secrets.")
            st.info("You can get an API key from: https://makersuite.google.com/app/apikey")
            return None
        
        # Create embeddings with proper error handling
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
        
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        
        # Provide helpful suggestions based on error type
        error_msg = str(e).lower()
        if "event loop" in error_msg or "asyncio" in error_msg:
            st.info("💡 **Solution**: This appears to be an asyncio event loop issue. Try:")
            st.code("pip install nest-asyncio", language="bash")
            st.write("Or consider using OpenAI embeddings as an alternative.")
        elif "api" in error_msg or "key" in error_msg:
            st.info("💡 **Solution**: Check your Google API key configuration.")
        elif "quota" in error_msg or "limit" in error_msg:
            st.info("💡 **Solution**: You may have exceeded your API quota. Check your Google Cloud console.")
            
        return None

def query_document(vector_store, question):
    """Query the document with a question."""
    try:
        # Fix asyncio event loop before querying
        fix_asyncio_loop()
        
        relevant_docs = vector_store.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Get API key with better error handling
        google_api_key = None
        
        if hasattr(st.session_state, 'google_api_key') and st.session_state.google_api_key:
            google_api_key = st.session_state.google_api_key
        elif os.getenv("GOOGLE_API_KEY"):
            google_api_key = os.getenv("GOOGLE_API_KEY")
        elif hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
            google_api_key = st.secrets["GOOGLE_API_KEY"]
        
        if not google_api_key:
            return "❌ Google API Key not found. Please configure your API key."
        
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.1
        )
        
        prompt = PromptTemplate(
            template="""Answer the question based on the following context. If the answer cannot be found in the context, say "I cannot find the answer in the document."

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        chain = prompt | model | StrOutputParser()
        response = chain.invoke({"context": context, "question": question})
        return response
        
    except Exception as e:
        st.error(f"❌ Error querying document: {str(e)}")
        return f"Sorry, I encountered an error while processing your question: {str(e)}"

def show_rag_page():
    """Display the RAG query system page."""
    st.header("🔍 RAG Query System")
    st.write("Upload a document and ask questions about it.")
    
    # Initialize session state variables if they don't exist
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'documents' not in st.session_state:
        st.session_state.documents = None
    
    # API Key input (optional - for users who want to input manually)
    with st.expander("🔑 API Configuration (Optional)", expanded=False):
        api_key_input = st.text_input(
            "Google API Key (optional - leave empty to use environment variables)",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        if api_key_input:
            st.session_state.google_api_key = api_key_input
            st.success("✅ API Key set!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file for RAG",
        type=['pdf', 'txt', 'docx', 'doc'],
        help="Upload a PDF, TXT, or DOCX file",
        key="rag_uploader"
    )
    
    if uploaded_file is not None:
        # Show file info
        st.info(f"📄 **File**: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        with st.spinner("Loading document..."):
            documents = load_file_for_rag(uploaded_file)
        
        if documents:
            st.success(f"✅ Successfully loaded {len(documents)} pages/sections")
            
            # Create vector store if not already created or if documents changed
            if (st.session_state.vector_store is None or 
                st.session_state.documents != documents):
                
                with st.spinner("Creating vector store (this may take a moment)..."):
                    vector_store = create_vector_store(documents)
                    
                    if vector_store is not None:
                        st.session_state.vector_store = vector_store
                        st.session_state.documents = documents
                        st.success("✅ Vector store created successfully!")
                    else:
                        st.error("❌ Failed to create vector store. Please check the error messages above.")
                        return
            
            # Query section
            st.subheader("💬 Ask Questions")
            
            # Show some example questions
            with st.expander("💡 Example Questions", expanded=False):
                st.write("""
                - What is the main topic of this document?
                - Can you summarize the key points?
                - What are the important dates mentioned?
                - Who are the main people/entities mentioned?
                """)
            
            question = st.text_input(
                "Enter your question about the document:",
                placeholder="What is this document about?"
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🔍 Ask Question", type="primary") and question.strip():
                    if st.session_state.vector_store is not None:
                        with st.spinner("🔍 Searching for answer..."):
                            answer = query_document(st.session_state.vector_store, question)
                        
                        st.subheader("💡 Answer")
                        st.write(answer)
                        
                        # Add to chat history if you want
                        if 'chat_history' not in st.session_state:
                            st.session_state.chat_history = []
                        
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': answer
                        })
                        
                    else:
                        st.error("❌ Vector store not available. Please upload a document first.")
                
                elif st.button("🔍 Ask Question", type="primary"):
                    st.warning("⚠️ Please enter a question.")
            
            with col2:
                if st.button("🗑️ Clear Document", type="secondary"):
                    st.session_state.vector_store = None
                    st.session_state.documents = None
                    if 'chat_history' in st.session_state:
                        st.session_state.chat_history = []
                    st.success("✅ Document cleared! Upload a new document to continue.")
                    st.rerun()
            
            # Show chat history
            if 'chat_history' in st.session_state and st.session_state.chat_history:
                st.subheader("📝 Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                    with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
                        st.write(f"**Question:** {chat['question']}")
                        st.write(f"**Answer:** {chat['answer']}")
    
    else:
        st.info("👆 Please upload a document to get started!")
        
        # Show supported formats
        st.subheader("📋 Supported File Formats")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("📄 **PDF Files**")
            st.write("- Research papers")
            st.write("- Reports")
            st.write("- Books")
        with col2:
            st.write("📝 **Text Files**")
            st.write("- Articles")
            st.write("- Notes")
            st.write("- Documentation")
        with col3:
            st.write("📋 **Word Documents**")
            st.write("- DOCX files")
            st.write("- DOC files")
            st.write("- Reports")
