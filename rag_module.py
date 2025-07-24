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

# Fix for asyncio event loop issues - improved approach
def run_async_in_thread(async_func, *args, **kwargs):
    """Run async function in a separate thread to avoid event loop conflicts."""
    import concurrent.futures
    import asyncio
    
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result()

def get_api_key():
    """Get Google API key from various sources with proper error handling."""
    api_key = None
    
    # Try session state first
    if hasattr(st.session_state, 'google_api_key') and st.session_state.google_api_key:
        api_key = st.session_state.google_api_key
    # Try Streamlit secrets
    elif hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    # Try environment variable
    elif os.getenv("GOOGLE_API_KEY"):
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        return None
    
    return api_key

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
    """Create FAISS vector store from documents with improved error handling."""
    try:
        # Get API key
        google_api_key = get_api_key()
        if not google_api_key:
            st.error("❌ Google API Key not found. Please set it in your environment variables or Streamlit secrets.")
            st.info("You can get an API key from: https://makersuite.google.com/app/apikey")
            return None
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            st.error("❌ No text chunks were created from the document.")
            return None
        
        st.info(f"📄 Created {len(chunks)} text chunks for processing")
        
        # Create embeddings - synchronous approach
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_api_key
            )
            
            # Create vector store with progress tracking
            with st.spinner("Creating embeddings... This may take a moment..."):
                vector_store = FAISS.from_documents(chunks, embeddings)
                
            return vector_store
            
        except Exception as embed_error:
            error_msg = str(embed_error).lower()
            st.error(f"❌ Error creating embeddings: {str(embed_error)}")
            
            # Provide specific error guidance
            if "quota" in error_msg or "limit" in error_msg:
                st.info("💡 **API Quota Issue**: You may have exceeded your API quota. Check your Google Cloud console.")
            elif "authentication" in error_msg or "api key" in error_msg:
                st.info("💡 **Authentication Issue**: Please verify your Google API key is correct and active.")
            elif "timeout" in error_msg:
                st.info("💡 **Timeout Issue**: Try with a smaller document or check your internet connection.")
            else:
                st.info("💡 **General Issue**: Try refreshing the page and uploading the document again.")
            
            return None
        
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return None

def query_document(vector_store, question):
    """Query the document with a question - improved version."""
    try:
        if not vector_store:
            return "❌ Vector store is not available. Please upload and process a document first."
        
        # Get API key
        google_api_key = get_api_key()
        if not google_api_key:
            return "❌ Google API Key not found. Please configure your API key."
        
        # Search for relevant documents
        try:
            relevant_docs = vector_store.similarity_search(question, k=3)
            if not relevant_docs:
                return "❌ No relevant content found in the document for your question."
        except Exception as search_error:
            st.error(f"Search error: {str(search_error)}")
            return "❌ Error occurred while searching the document."
        
        # Combine context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        if not context.strip():
            return "❌ No relevant content found to answer your question."
        
        # Create model
        try:
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=google_api_key,
                temperature=0.1,
                max_output_tokens=1024
            )
        except Exception as model_error:
            st.error(f"Model creation error: {str(model_error)}")
            return "❌ Error creating the AI model."
        
        # Create prompt
        prompt = PromptTemplate(
            template="""You are a helpful assistant that answers questions based on the provided context.

Context from the document:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the answer cannot be found in the context, say "I cannot find the answer in the provided document."
- Be concise and accurate
- If you're not sure, say so

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Create chain and get response
        try:
            chain = prompt | model | StrOutputParser()
            response = chain.invoke({"context": context, "question": question})
            return response
        except Exception as chain_error:
            st.error(f"Chain execution error: {str(chain_error)}")
            return f"❌ Error processing your question: {str(chain_error)}"
        
    except Exception as e:
        st.error(f"❌ Unexpected error in query_document: {str(e)}")
        return f"❌ Sorry, I encountered an error while processing your question: {str(e)}"

def show_rag_page():
    """Display the RAG query system page."""
    st.header("🔍 RAG Query System")
    st.write("Upload a document and ask questions about it.")
    
    # Initialize session state variables
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'documents' not in st.session_state:
        st.session_state.documents = None
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    
    # API Key configuration
    with st.expander("🔑 API Configuration (Optional)", expanded=False):
        api_key_input = st.text_input(
            "Google API Key (optional - leave empty to use environment variables)",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        if api_key_input:
            st.session_state.google_api_key = api_key_input
            st.success("✅ API Key set!")
        
        # Show current API key status
        current_key = get_api_key()
        if current_key:
            st.success("✅ API Key is configured")
        else:
            st.warning("⚠️ No API Key found. Please set one above or in your environment.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file for RAG",
        type=['pdf', 'txt', 'docx', 'doc'],
        help="Upload a PDF, TXT, or DOCX file",
        key="rag_uploader"
    )
    
    if uploaded_file is not None:
        # Show file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📄 **File**: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        # Check if we need to process this file
        need_processing = (
            st.session_state.vector_store is None or 
            st.session_state.processed_file != uploaded_file.name
        )
        
        if need_processing:
            with st.spinner("Loading document..."):
                documents = load_file_for_rag(uploaded_file)
            
            if documents:
                st.success(f"✅ Successfully loaded {len(documents)} pages/sections")
                
                # Create vector store
                with st.spinner("Creating vector store (this may take a moment)..."):
                    vector_store = create_vector_store(documents)
                    
                    if vector_store is not None:
                        st.session_state.vector_store = vector_store
                        st.session_state.documents = documents
                        st.session_state.processed_file = uploaded_file.name
                        st.success("✅ Vector store created successfully! You can now ask questions.")
                    else:
                        st.error("❌ Failed to create vector store. Please check the error messages above.")
                        return
            else:
                st.error("❌ Failed to load the document. Please try a different file.")
                return
        else:
            st.success("✅ Document already processed and ready for questions!")
        
        # Query section
        if st.session_state.vector_store is not None:
            st.subheader("💬 Ask Questions")
            
            # Example questions
            with st.expander("💡 Example Questions", expanded=False):
                st.write("""
                **Try asking:**
                - What is the main topic of this document?
                - Can you summarize the key points?
                - What are the important dates mentioned?
                - Who are the main people/entities mentioned?
                - What conclusions does the document reach?
                """)
            
            question = st.text_input(
                "Enter your question about the document:",
                placeholder="What is this document about?",
                key="question_input"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                ask_button = st.button("🔍 Ask Question", type="primary")
            
            with col2:
                if st.button("🗑️ Clear Document", type="secondary"):
                    st.session_state.vector_store = None
                    st.session_state.documents = None
                    st.session_state.processed_file = None
                    if 'chat_history' in st.session_state:
                        st.session_state.chat_history = []
                    st.success("✅ Document cleared!")
                    st.rerun()
            
            with col3:
                if st.button("🔄 Reset Chat", type="secondary"):
                    if 'chat_history' in st.session_state:
                        st.session_state.chat_history = []
                    st.success("✅ Chat history cleared!")
            
            # Process question
            if ask_button and question.strip():
                with st.spinner("🔍 Searching for answer..."):
                    answer = query_document(st.session_state.vector_store, question)
                
                st.subheader("💡 Answer")
                st.write(answer)
                
                # Add to chat history
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': answer
                })
                
                # Clear the input
                st.session_state.question_input = ""
                
            elif ask_button:
                st.warning("⚠️ Please enter a question.")
            
            # Show chat history
            if 'chat_history' in st.session_state and st.session_state.chat_history:
                st.subheader("📝 Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                    with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
                        st.write(f"**Question:** {chat['question']}")
                        st.write(f"**Answer:** {chat['answer']}")
    
    else:
        st.info("👆 Please upload a document to get started!")
        
        # Show supported formats and tips
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Supported File Formats")
            st.write("📄 **PDF Files** - Research papers, reports, books")
            st.write("📝 **Text Files** - Articles, notes, documentation")
            st.write("📋 **Word Documents** - DOCX and DOC files")
        
        with col2:
            st.subheader("💡 Tips for Best Results")
            st.write("✅ Use documents with clear, readable text")
            st.write("✅ Smaller files (< 50MB) process faster")
            st.write("✅ Ask specific questions for better answers")
            st.write("✅ Try different phrasings if you don't get good results")
