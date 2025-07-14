import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

def load_file(file_path):
    """Load document from various file formats."""
    file_extension = file_path.lower().split('.')[-1]
    
    if file_extension == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == 'txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_extension in ['docx', 'doc']:
        loader = Docx2txtLoader(file_path)
    else:
        print(f"Unsupported file format: {file_extension}")
        return None
    
    documents = loader.load()
    print(f"Loaded {len(documents)} pages/sections from {file_path}")
    return documents

def create_vector_store(documents):
    """Create FAISS vector store from documents."""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def query_document(vector_store, question):
    """Query the document with a question."""
    # Get relevant chunks
    relevant_docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Create response using Gemini
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
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

def main():
    # Get file path from user
    file_path = input("XG BOOST OFFICIAL RESEARCH PAPER.pdf")
    
    # Load the document
    documents = load_file(file_path)
    if not documents:
        print("Failed to load document. Please check the file path and format.")
        return
    
    # Create vector store
    print("Creating vector store...")
    vector_store = create_vector_store(documents)
    print("Vector store created successfully!")
    
    # Query loop
    print("\nYou can now ask questions about your document. Type 'quit' to exit.")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'quit':
            break
        
        answer = query_document(vector_store, question)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
