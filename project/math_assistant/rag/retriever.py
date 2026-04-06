import os
import re
import hashlib
import logging
from typing import List, Dict

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        class Document:
            def __init__(self, page_content: str, metadata: dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}

logger = logging.getLogger("math_assistant")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

class MathDataLoader:
    def __init__(self):
        self.documents = []

    def load_builtin_knowledge(self, data_dir: str = None):
        """Load text files from the data directory"""
        if data_dir is None:
            # Assume data/ is parallel to rag/
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
        
        docs = []
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith(".txt"):
                    filepath = os.path.join(data_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                            topic = filename.replace(".txt", "")
                            docs.append(Document(page_content=content, metadata={"source": "knowledge_base", "topic": topic}))
                    except Exception as e:
                        logger.error(f"Failed to read {filepath}: {e}")
        logger.info(f"Loading {len(docs)} built-in knowledge documents from {data_dir}")
        return docs

    def load_pdf(self, pdf_path: str):
        try:
            from langchain_community.document_loaders import PyPDFLoader
            docs = PyPDFLoader(pdf_path).load()
            logger.info(f"Loaded {len(docs)} pages from: {pdf_path}")
            return docs
        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_path}: {e}")
            return []

    def load_pdfs_from_directory(self, dir_path: str):
        try:
            from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
            docs = DirectoryLoader(dir_path, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
            logger.info(f"Loaded {len(docs)} documents from: {dir_path}")
            return docs
        except Exception as e:
            logger.error(f"Failed to load PDFs from {dir_path}: {e}")
            return []

    def load_web_pages(self, urls: List[str]):
        from langchain_community.document_loaders import WebBaseLoader
        docs = []
        for url in urls:
            try:
                docs.extend(WebBaseLoader(url).load())
                logger.info(f"Loaded: {url}")
            except Exception as e:
                logger.warning(f"Failed URL {url}: {e}")
        return docs

    def load_text_file(self, file_path: str):
        try:
            if file_path.endswith(".md"):
                from langchain_community.document_loaders import UnstructuredMarkdownLoader
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            logger.info(f"Loaded: {file_path}")
            return docs
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []

    def load_all(self, pdf_paths=None, urls=None, text_paths=None, pdf_directory=None):
        all_docs = self.load_builtin_knowledge()
        if pdf_paths:
            for p in pdf_paths: all_docs.extend(self.load_pdf(p))
        if pdf_directory and os.path.exists(pdf_directory):
            all_docs.extend(self.load_pdfs_from_directory(pdf_directory))
        if urls:
            all_docs.extend(self.load_web_pages(urls))
        if text_paths:
            for p in text_paths: all_docs.extend(self.load_text_file(p))
        logger.info(f"Total documents loaded: {len(all_docs)}")
        self.documents = all_docs
        return all_docs


class MathDataPreprocessor:
    TOPIC_KEYWORDS: Dict[str, List[str]] = {
        "calculus":       ["derivative", "integral", "differentiate", "integrate", "limit", "continuity", "taylor"],
        "linear_algebra": ["matrix", "vector", "eigenvalue", "determinant", "rank", "span", "basis"],
        "statistics":     ["probability", "distribution", "mean", "variance", "regression", "hypothesis"],
        "algebra":        ["polynomial", "equation", "quadratic", "factor", "root", "logarithm", "exponent"],
        "trigonometry":   ["sine", "cosine", "tangent", "angle", "radian", "unit circle", "trig"],
        "discrete_math":  ["graph", "combinatorics", "permutation", "combination", "modular", "prime"],
        "geometry":       ["triangle", "circle", "area", "volume", "perimeter", "pythagorean", "coordinate"],
        "number_theory":  ["prime", "divisor", "gcd", "lcm", "modular", "congruence", "integer"],
    }

    def __init__(self):
        self._seen_hashes: set = set()

    def _clean(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"https?://\S+", "[URL]", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        replacements = {
            "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "--", "\u00a0": " ",
            "\u03c0": "pi", "\u221e": "infinity",
            "\u2264": "<=", "\u2265": ">="
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.strip()

    def _detect_topic(self, text: str) -> str:
        tl = text.lower()
        scores = {t: sum(1 for kw in kws if kw in tl) for t, kws in self.TOPIC_KEYWORDS.items()}
        scores = {k: v for k, v in scores.items() if v > 0}
        return max(scores, key=scores.get) if scores else "general_math"

    def _difficulty(self, text: str) -> str:
        adv = ["eigenvalue", "differential equation", "fourier", "laplace", "manifold", "tensor"]
        mid = ["derivative", "integral", "matrix", "probability", "polynomial", "logarithm"]
        tl = text.lower()
        if sum(1 for t in adv if t in tl) >= 2: return "advanced"
        if sum(1 for t in mid if t in tl) >= 2: return "intermediate"
        return "beginner"

    def preprocess_document(self, doc):
        text = doc.page_content
        if len(text.strip()) < 50:
            return None
        text = self._clean(text)
        h = hashlib.md5(text.strip().lower().encode()).hexdigest()
        if h in self._seen_hashes:
            return None
        self._seen_hashes.add(h)
        meta = doc.metadata.copy()
        meta.update({
            "topic":        meta.get("topic") or self._detect_topic(text),
            "difficulty":   self._difficulty(text),
            "char_count":   len(text),
            "word_count":   len(text.split()),
            "content_hash": h,
        })
        return Document(page_content=text, metadata=meta)

    def preprocess_documents(self, documents):
        logger.info(f"Preprocessing {len(documents)} documents...")
        self._seen_hashes.clear()
        processed = [r for doc in documents if (r := self.preprocess_document(doc)) is not None]
        logger.info(f"Done: {len(processed)} kept, {len(documents)-len(processed)} skipped")
        return processed


class MathTextSplitter:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""])
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")])

    def split_document(self, doc):
        source = doc.metadata.get("source", "").lower()
        if source.endswith(".md") or "markdown" in source:
            try:
                splits = self.markdown_splitter.split_text(doc.page_content)
                chunks = [Document(page_content=s.page_content,
                                   metadata={**doc.metadata, **s.metadata}) for s in splits]
            except Exception:
                chunks = self.recursive.split_documents([doc])
        else:
            chunks = self.recursive.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "chunk_size":   len(chunk.page_content),
            })
        return chunks

    def split_documents(self, documents):
        logger.info(f"Splitting {len(documents)} documents...")
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        logger.info(f"Created {len(all_chunks)} chunks")
        return all_chunks
