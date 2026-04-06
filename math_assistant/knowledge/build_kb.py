import logging
from math_assistant.rag.retriever import MathDataLoader, MathDataPreprocessor, MathTextSplitter
from math_assistant.rag.vector_store import MathVectorStore

logger = logging.getLogger("math_assistant")

def build_pipeline(pdf_paths=None, urls=None, text_paths=None, force_rebuild=False) -> MathVectorStore:
    store = MathVectorStore()
    if store.is_ready() and not force_rebuild:
        logger.info(f"Knowledge base already built ({store.get_document_count()} docs).")
        return store
    raw_docs   = MathDataLoader().load_all(pdf_paths=pdf_paths or [], urls=urls or [], text_paths=text_paths or [])
    clean_docs = MathDataPreprocessor().preprocess_documents(raw_docs)
    chunks     = MathTextSplitter().split_documents(clean_docs)
    store.build_knowledge_base(chunks)
    return store
