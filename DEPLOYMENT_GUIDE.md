# 🚀 Streamlit Deployment Guide

This guide will help you deploy your Personal College Assistant app on Streamlit Cloud securely without exposing your API keys.

## 📋 Prerequisites

1. **GitHub Account** - Your code needs to be on GitHub
2. **Streamlit Cloud Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Google AI API Key** - Get one from [Google AI Studio](https://makersuite.google.com/app/apikey)

## 🔧 Step 1: Prepare Your Local Environment

### 1.1 Set up API key for local development
You have two options for local development:

**Option A: Use .env file (Recommended for local development)**
Create a `.env` file in your project root:
```bash
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

**Option B: Use Streamlit secrets**
Edit `.streamlit/secrets.toml` and replace the placeholder:
```toml
GOOGLE_API_KEY = "your_actual_gemini_api_key_here"
```

### 1.2 Verify .gitignore
Make sure both `.env` and `.streamlit/secrets.toml` are in your `.gitignore` file to prevent them from being committed to GitHub.

## 📤 Step 2: Push to GitHub

1. **Add your files to git:**
   ```bash
   git add .
   git commit -m "Add Streamlit app with secure secrets management"
   git push origin main
   ```

2. **Verify secrets.toml is NOT in your repository:**
   - Check that `.streamlit/secrets.toml` is not visible on GitHub
   - The file should be ignored due to `.gitignore`

## 🌐 Step 3: Deploy on Streamlit Cloud

### 3.1 Create New App
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"

### 3.2 Configure App Settings
- **Repository**: Select your GitHub repository
- **Branch**: `main` (or your default branch)
- **Main file path**: `app.py`
- **App URL**: Choose a custom URL (optional)

### 3.3 Add Secrets
1. Click "Advanced settings" or "Edit secrets"
2. Add your API key in the secrets editor:

```toml
GOOGLE_API_KEY = "your_actual_gemini_api_key_here"
```

3. Click "Save"

### 3.4 Deploy
1. Click "Deploy!"
2. Wait for the build to complete
3. Your app will be live at the provided URL

## 🔒 Security Features

✅ **API Key Protection**: Your API key is stored securely in Streamlit Cloud  
✅ **No Public Exposure**: The key never appears in your code or GitHub  
✅ **Environment Isolation**: Each deployment has its own secure environment  

## 🧪 Testing Your Deployment

1. **Upload a document** (PDF, TXT, or DOCX)
2. **Test summarization** - Try both brief and detailed summaries
3. **Test RAG queries** - Ask questions about your uploaded document
4. **Verify functionality** - All features should work as expected

## 🛠️ Troubleshooting

### Common Issues:

1. **"API Key not found" error**
   - Check that you added the secret in Streamlit Cloud
   - Verify the key name is exactly `GOOGLE_API_KEY`

2. **Build fails**
   - Check that all dependencies are in `requirements.txt`
   - Verify your main file path is correct

3. **App loads but features don't work**
   - Check your API key is valid
   - Verify you have sufficient API quota

### Support:
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Community](https://discuss.streamlit.io/)

## 📝 File Structure After Deployment

```
your-repo/
├── app.py                    # Main Streamlit app
├── summarizer_module.py      # Document summarization
├── rag_module.py            # RAG query system
├── requirements.txt         # Dependencies
├── .gitignore              # Git ignore rules
├── .streamlit/
│   └── secrets.toml        # Local secrets (not in GitHub)
└── DEPLOYMENT_GUIDE.md     # This guide
```

## 🎉 Success!

Your Personal College Assistant is now live and secure! Students can:
- Upload documents and get AI-powered summaries
- Ask questions about their documents using RAG
- Access the app from anywhere with an internet connection

The app is now ready for production use! 🚀 