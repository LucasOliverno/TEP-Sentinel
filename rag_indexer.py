"""Build the ChromaDB index from the domain knowledge base and the fault catalogue."""
import logging
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DIR, FAULT_DESCRIPTIONS, KB_DIR
from rag_agent import EMBEDDING_MODEL

load_dotenv()
log = logging.getLogger("tep.indexer")

# Only domain knowledge is indexed; project status reports would pollute retrieval.
DOMAIN_DOCS = ["contexto_tep.md", "dicionario_de_dados_tep.md"]


def build_documents():
    docs = []
    for name in DOMAIN_DOCS:
        docs.extend(TextLoader(str(KB_DIR / name), encoding="utf-8").load())
    for idv, text in FAULT_DESCRIPTIONS.items():
        docs.append(Document(page_content=f"IDV({idv}): {text}", metadata={"source": "fault_catalogue", "idv": idv}))
    return docs


def index_knowledge_base():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not found. Set it in .env")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,
                                              separators=["\n## ", "\n### ", "\n", " ", ""])
    splits = splitter.split_documents(build_documents())
    log.info("Indexing %d chunks with %s", len(splits), EMBEDDING_MODEL)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    store = Chroma.from_documents(splits, embeddings, persist_directory=str(CHROMA_DIR))
    log.info("Done: %d vectors in %s", store._collection.count(), CHROMA_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    index_knowledge_base()
