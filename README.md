# 🎓 Personal College Assistant

A powerful AI-powered Streamlit web application designed to help students efficiently process and understand their academic documents through intelligent summarization and interactive Q&A capabilities.

## 🌟 Overview

The Personal College Assistant leverages Google's Gemini AI to provide two core functionalities:
- **Document Summarization**: Generate brief or detailed summaries of uploaded documents
- **RAG Query System**: Ask questions about document content and receive contextual answers

## ✨ Features

### 📄 Document Summarizer
- **Multi-format Support**: PDF, TXT, DOCX, and DOC files
- **Flexible Summary Types**: 
  - Brief summaries (under 150 words) for quick overviews
  - Detailed summaries with methodology, findings, and insights
- **Structured Output**: Organized format with main topics, key points, and conclusions

### 🔍 RAG (Retrieval-Augmented Generation) Query System
- **Interactive Q&A**: Ask questions about uploaded documents
- **Contextual Responses**: AI answers based on document content
- **Vector Search**: Efficient similarity search using FAISS
- **Session Memory**: Maintains document context during the session

### 🔐 Security & Deployment
- **Secure API Key Management**: Uses Streamlit secrets for safe deployment
- **Cloud-Ready**: Optimized for Streamlit Cloud deployment
- **Environment Isolation**: Separate development and production configurations

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd personal-college-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key**
   
   Create `.streamlit/secrets.toml`:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key_here"
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   
   Open your browser and go to `http://localhost:8501`

## 📁 Project Structure

```
personal-college-assistant/
├── app.py                      # Main Streamlit application
├── summarizer_module.py        # Document summarization logic
├── rag_module.py              # RAG query system implementation
├── requirements.txt           # Python dependencies
├── DEPLOYMENT_GUIDE.md        # Detailed deployment instructions
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
├── .devcontainer/
│   └── devcontainer.json      # VS Code dev container config
└── .streamlit/
    └── secrets.toml           # Local API key configuration
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **AI Model** | Google Gemini (via LangChain) |
| **Vector Database** | FAISS |
| **Document Processing** | PyPDF, docx2txt, python-docx |
| **Embeddings** | Google Generative AI Embeddings |
| **Framework** | LangChain |

## 📖 Usage Guide

### Document Summarization

1. **Upload Document**: Choose a PDF, TXT, DOCX, or DOC file
2. **Select Summary Type**: 
   - **Brief**: Concise overview with main points
   - **Detailed**: Comprehensive analysis with methodology and findings
3. **Generate Summary**: Click to process and receive structured summary

### RAG Query System

1. **Upload Document**: Same file types as summarization
2. **Wait for Processing**: Vector store creation (one-time per document)
3. **Ask Questions**: Type natural language questions about the document
4. **Get Answers**: Receive contextual responses based on document content

## 🔧 Configuration

### Environment Variables
The application uses the following configuration:

```toml
# .streamlit/secrets.toml
GOOGLE_API_KEY = "your_gemini_api_key_here"
```

### Supported File Formats
- **PDF**: `.pdf`
- **Text**: `.txt`
- **Word Documents**: `.docx`, `.doc`

### Processing Limits
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Vector Search**: Top 3 relevant chunks

## 🚀 Deployment

### Local Development
Follow the Quick Start guide above for local setup.

### Production Deployment
For detailed deployment instructions to Streamlit Cloud, see [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md).

**Quick Deploy Steps:**
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add API key to secrets
4. Deploy!

## 📋 Dependencies

### Core Dependencies
```
streamlit                 # Web framework
langchain-google-genai   # Google AI integration
langchain               # LLM framework
langchain-community     # Community integrations
faiss-cpu              # Vector similarity search
```

### Document Processing
```
pypdf                   # PDF processing
docx2txt               # Word document processing
python-dotenv          # Environment variable management
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

**API Key Errors**
- Ensure your Google Gemini API key is correctly set in `.streamlit/secrets.toml`
- Verify the key has sufficient quota and permissions

**File Upload Issues**
- Check file format is supported (PDF, TXT, DOCX, DOC)
- Ensure file size is reasonable (< 200MB recommended)

**Vector Store Errors**
- Clear session state and re-upload document
- Check available memory for large documents

### Getting Help

- Check the [Deployment Guide](DEPLOYMENT_GUIDE.md) for deployment issues
- Review error messages in the Streamlit interface
- Ensure all dependencies are properly installed

## 🎯 Roadmap

- [ ] Support for additional file formats (PPT, Excel)
- [ ] Batch document processing
- [ ] Citation tracking in responses
- [ ] Document comparison features
- [ ] Export functionality for summaries

## 📊 Performance

- **Response Time**: < 5 seconds for most queries
- **Memory Usage**: Optimized for standard cloud deployments
- **File Size**: Supports documents up to 200MB
- **Concurrent Users**: Scales with Streamlit Cloud resources

## 🙏 Acknowledgments

- Google for the Gemini AI API
- LangChain for the powerful framework
- Streamlit for the amazing web framework
- FAISS for efficient vector similarity search

---

**Made with ❤️ for students everywhere**

*Transform your study experience with AI-powered document intelligence!*
