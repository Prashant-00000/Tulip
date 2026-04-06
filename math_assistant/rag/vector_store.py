import os
import logging
from pathlib import Path
from math_assistant.rag.embeddings import get_embeddings

logger = logging.getLogger("math_assistant")
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma").lower()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
COLLECTION_NAME = "math_knowledge_base"

class MathVectorStore:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vectorstore = None
        self.db_type = VECTOR_DB_TYPE
        self._load_existing()

    def _load_existing(self):
        if self.db_type == "chroma":
            self._try_chroma()
        else:
            self._try_faiss()

    def _try_chroma(self, documents=None):
        try:
            from langchain_community.vectorstores import Chroma
            persist_path = Path(CHROMA_PERSIST_DIR)
            persist_path.mkdir(parents=True, exist_ok=True)
            if documents:
                self.vectorstore = Chroma.from_documents(
                    documents=documents, embedding=self.embeddings,
                    collection_name=COLLECTION_NAME,
                    persist_directory=str(persist_path))
                logger.info("ChromaDB created.")
            elif list(persist_path.glob("*.sqlite3")):
                self.vectorstore = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=self.embeddings,
                    persist_directory=str(persist_path))
                logger.info(f"ChromaDB loaded ({self.vectorstore._collection.count()} docs)")
        except Exception as e:
            logger.warning(f"ChromaDB failed ({e}), switching to FAISS")
            self.db_type = "faiss"
            if documents:
                self._try_faiss(documents)

    def _try_faiss(self, documents=None):
        try:
            from langchain_community.vectorstores import FAISS
            index_path = Path(FAISS_INDEX_PATH)
            if documents:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                index_path.mkdir(parents=True, exist_ok=True)
                self.vectorstore.save_local(str(index_path))
                logger.info(f"FAISS saved to {index_path}")
            elif index_path.exists() and any(index_path.iterdir()):
                self.vectorstore = FAISS.load_local(
                    str(index_path), self.embeddings,
                    allow_dangerous_deserialization=True)
                logger.info("FAISS loaded.")
        except Exception as e:
            logger.error(f"FAISS failed: {e}")

    def build_knowledge_base(self, documents):
        logger.info(f"Building knowledge base with {len(documents)} chunks...")
        if self.db_type == "chroma":
            self._try_chroma(documents)
        else:
            self._try_faiss(documents)
        logger.info("Knowledge base ready.")

    def add_documents(self, documents):
        if self.vectorstore is None:
            self.build_knowledge_base(documents)
        else:
            self.vectorstore.add_documents(documents)

    def similarity_search(self, query: str, k: int = TOP_K_RESULTS, filter_topic: str = None):
        if self.vectorstore is None:
            return []
        try:
            if filter_topic and self.db_type == "chroma":
                return self.vectorstore.similarity_search(query, k=k, filter={"topic": filter_topic})
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def as_retriever(self, k: int = TOP_K_RESULTS):
        return self.vectorstore.as_retriever(search_kwargs={"k": k}) if self.vectorstore else None

    def get_document_count(self) -> int:
        if self.vectorstore is None:
            return 0
        try:
            return (self.vectorstore._collection.count() if self.db_type == "chroma"
                    else self.vectorstore.index.ntotal)
        except Exception:
            return 0

    def is_ready(self) -> bool:
        return self.vectorstore is not None and self.get_document_count() > 0
