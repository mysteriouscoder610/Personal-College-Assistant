import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import tempfile
import os

def load_file_for_summary(uploaded_file):
    """Load document from uploaded file for summarization."""
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

def summarize_document(documents, summary_type):
    """Summarize document using the specified type."""
    # Get API key from session state
    google_api_key = st.session_state.google_api_key
    
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key
    )
    
    if summary_type == "Brief":
        prompt = PromptTemplate(
            template = '''Create a BRIEF summary with this format:
                **MAIN TOPIC**: [One sentence]
                **KEY POINTS**: [3-4 bullet points]
                **CONCLUSION**: [One sentence takeaway]

                Requirements:
                - Be 100 percent accurate to the source
                - Keep it very concise (under 150 words)
                - Focus on the most essential information
                - {text}
                ''',
            input_variables = ['text']
        )
    else:  # Detailed
        prompt = PromptTemplate(
            template = '''Create a DETAILED summary with this format:
                **MAIN TOPIC**: [One sentence]
                **KEY POINTS**: [5-7 bullet points]
                **CRITICAL INSIGHTS**: [Most important discoveries]
                **METHODOLOGY**: [How the research was conducted]
                **FINDINGS**: [Key results and data]
                **CONCLUSION**: [Final takeaway and implications]

                Requirements:
                - Be 100 percent accurate to the source
                - Use engaging formatting
                - Highlight what matters most
                - Make it visually appealing
                - Be comprehensive and thorough
                - {text}
                ''',
            input_variables = ['text']
        )
    
    parser = StrOutputParser()
    chain = prompt | model | parser
    
    # Combine all document content
    full_text = "\n\n".join([doc.page_content for doc in documents])
    
    return chain.invoke({"text": full_text})

def show_summarizer_page():
    """Display the document summarizer page."""
    st.header("📄 Document Summarizer")
    st.write("Upload a document and get a brief or detailed summary.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'docx', 'doc'],
        help="Upload a PDF, TXT, or DOCX file"
    )
    
    if uploaded_file is not None:
        with st.spinner("Loading document..."):
            documents = load_file_for_summary(uploaded_file)
        
        if documents:
            st.success(f"✅ Successfully loaded {len(documents)} pages/sections")
            
            # Summary type selection
            summary_type = st.radio(
                "Choose summary type:",
                ["Brief", "Detailed"],
                horizontal=True
            )
            
            if st.button("Generate Summary", type="primary"):
                with st.spinner(f"Generating {summary_type.lower()} summary..."):
                    summary = summarize_document(documents, summary_type)
                
                st.subheader(f"{summary_type} Summary")
                st.markdown(summary) 