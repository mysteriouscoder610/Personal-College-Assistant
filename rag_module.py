import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
import tempfile
import os

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Get API key from session state or directly from environment
    try:
        google_api_key = st.session_state.google_api_key
    except (KeyError, AttributeError):
        # Fallback to environment variable if not in session state
        google_api_key = os.getenv("GOOGLE_API_KEY")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def query_document(vector_store, question):
    """Query the document with a question."""
    relevant_docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Get API key from session state or directly from environment
    try:
        google_api_key = st.session_state.google_api_key
    except (KeyError, AttributeError):
        # Fallback to environment variable if not in session state
        google_api_key = os.getenv("GOOGLE_API_KEY")
    
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key
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

def show_rag_page():
    """Display the RAG query system page."""
    st.header("🔍 RAG Query System")
    st.write("Upload a document and ask questions about it.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file for RAG",
        type=['pdf', 'txt', 'docx', 'doc'],
        help="Upload a PDF, TXT, or DOCX file",
        key="rag_uploader"
    )
    
    if uploaded_file is not None:
        with st.spinner("Loading document..."):
            documents = load_file_for_rag(uploaded_file)
        
        if documents:
            st.success(f"✅ Successfully loaded {len(documents)} pages/sections")
            
            # Create vector store if not already created
            if st.session_state.vector_store is None:
                with st.spinner("Creating vector store..."):
                    st.session_state.vector_store = create_vector_store(documents)
                    st.session_state.documents = documents
                st.success("✅ Vector store created successfully!")
            
            # Query section
            st.subheader("Ask Questions")
            question = st.text_input("Enter your question about the document:")
            
            if st.button("Ask Question", type="primary") and question:
                with st.spinner("Searching for answer..."):
                    answer = query_document(st.session_state.vector_store, question)
                
                st.subheader("Answer")
                st.write(answer)
            
            # Clear vector store
            if st.button("Clear Document", type="secondary"):
                st.session_state.vector_store = None
                st.session_state.documents = None
                st.success("Document cleared! Upload a new document to continue.")
                st.rerun() 