# 🎓 Personal College Assistant

A powerful Streamlit web application that provides AI-powered document summarization and RAG (Retrieval-Augmented Generation) query capabilities for students.

## ✨ Features

- **📄 Document Summarizer**: Upload documents and get brief or detailed AI summaries
- **🔍 RAG Query System**: Ask questions about your uploaded documents and get contextual answers
- **📁 Multi-format Support**: PDF, TXT, DOCX, and DOC files
- **🔐 Secure API Key Management**: Uses Streamlit secrets for secure deployment
- **⚡ Fast Processing**: Optimized document chunking and vector search

## Requirements

- Python 3.11+
- Google Gemini API key
- Required packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
   - Create `.streamlit/secrets.toml` file
   - Add your Google Gemini API key:
   ```toml
   GOOGLE_API_KEY = "your_api_key_here"
   ```

## Usage

Run the application:
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### Document Summarizer
1. Upload a document (PDF, TXT, DOCX, DOC)
2. Select summary type (Brief or Detailed)
3. Generate summary

### RAG Query System
1. Upload a document
2. Ask questions about the document content
3. Get AI-powered answers based on the document

## Deployment

Deploy to Streamlit Cloud following the instructions in `DEPLOYMENT_GUIDE.md`.

## File Structure

```
├── app.py                    # Main Streamlit application
├── summarizer_module.py      # Document summarization functionality
├── rag_module.py            # RAG query system
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── .streamlit/
│   └── secrets.toml        # API key configuration
└── .devcontainer/
    └── devcontainer.json   # Development container config
```

## Dependencies

- streamlit
- langchain-google-genai
- langchain
- langchain-community
- faiss-cpu
- pypdf
- docx2txt
- python-dotenv

## 🚀 Getting Started

1. **Clone** the repository
2. **Install** dependencies with `pip install -r requirements.txt`
3. **Add** your Google Gemini API key to `.streamlit/secrets.toml`
4. **Run** with `streamlit run app.py`
5. **Upload** documents and start exploring!

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Models**: Google Gemini (via LangChain)
- **Vector Store**: FAISS
- **Document Processing**: PyPDF, docx2txt
- **Embeddings**: Google Generative AI Embeddings

## 📝 Supported File Types

- 📄 PDF files
- 📝 Text files (.txt)
- 📋 Word documents (.docx, .doc)

## 🔧 Configuration

The app uses Streamlit secrets for secure API key management. Make sure to add your Google Gemini API key to `.streamlit/secrets.toml` before running locally.

## 🌐 Deployment

Ready to deploy? Check out the detailed `DEPLOYMENT_GUIDE.md` for step-by-step instructions to deploy on Streamlit Cloud.

## ⚡ Performance Tips

- For large documents, the app automatically chunks text for optimal processing
- Vector store is cached in session state for faster subsequent queries
- Brief summaries are under 150 words for quick reading
