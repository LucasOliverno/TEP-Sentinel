
import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Standard
# import google.generativeai as genai # Removed direct dependency
from dotenv import load_dotenv

# Load Env
load_dotenv()

# Setup Paths
KB_DIR = "Banco de Conhecimento"
DB_DIR = "chroma_db"

def index_knowledge_base():
    # 1. Verify API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found.")
        return
    
    print(f"Indexing Knowledge Base from '{KB_DIR}'...")
    
    # 2. Load Documents
    documents = []
    files = glob.glob(f"{KB_DIR}/*.md")
    
    for f in files:
        print(f"Loading {f}...")
        try:
            loader = TextLoader(f, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    print(f"Total documents loaded: {len(documents)}")

    # 3. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks.")

    # 4. Create Embeddings & Store in Chroma
    print("Generating Embeddings and indexing in ChromaDB...")
    
    # Standard LangChain Class
    print("Using Embedding Model: models/gemini-embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=api_key
    )
    
    # Persist directly
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    
    print(f"Indexing Complete. Vector DB saved to '{DB_DIR}'.")
    # Verify by counting
    print(f"Collection count: {vectorstore._collection.count()}")

if __name__ == "__main__":
    index_knowledge_base()
