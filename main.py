"""
╔══════════════════════════════════════════════════════════════════════╗
║          ADVANCED MATHEMATICS ASSISTANT — main.py                   ║
║  All 7 pipeline steps in one file                                   ║
║                                                                      ║
║  USAGE:                                                              ║
║    python3.11 -m streamlit run main.py   → Launch UI                ║
║    python3.11 main.py --setup            → Build knowledge base     ║
║    python3.11 main.py --test             → Run all tests            ║
║    python3.11 main.py --eval             → Evaluate RAG pipeline    ║
║    python3.11 main.py --rebuild          → Force rebuild KB         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, re, sys, json, time, uuid, hashlib, logging, argparse, unittest
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("math_assistant")

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
LLM_MODEL          = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DB_TYPE     = os.getenv("VECTOR_DB_TYPE", "chroma").lower()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
FAISS_INDEX_PATH   = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
TOP_K_RESULTS      = int(os.getenv("TOP_K_RESULTS", "5"))
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "200"))
MONGODB_URI        = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME    = os.getenv("MONGODB_DB_NAME", "math_assistant")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chat_history")
COLLECTION_NAME    = "math_knowledge_base"

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

try:
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError:
    from langchain.schema import HumanMessage, AIMessage

try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — DATA SOURCES                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

MATH_KNOWLEDGE_BASE = [
    Document(page_content="""Calculus Fundamentals:
The derivative of f(x) measures the rate of change. Key rules:
- Power Rule: d/dx[x^n] = n*x^(n-1)
- Product Rule: d/dx[f*g] = f'g + fg'
- Chain Rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)
- Quotient Rule: d/dx[f/g] = (f'g - fg') / g^2
Common derivatives:
- d/dx[sin(x)] = cos(x)
- d/dx[cos(x)] = -sin(x)
- d/dx[e^x] = e^x
- d/dx[ln(x)] = 1/x
Integration (antiderivative):
- integral(x^n dx) = x^(n+1)/(n+1) + C
- integral(e^x dx) = e^x + C
- integral(sin(x) dx) = -cos(x) + C
- integral(cos(x) dx) = sin(x) + C
Fundamental Theorem of Calculus:
integral[a to b] f(x) dx = F(b) - F(a)  where F'(x) = f(x)""",
    metadata={"source": "knowledge_base", "topic": "calculus"}),

    Document(page_content="""Linear Algebra Essentials:
Matrix Operations:
- Matrix multiplication (A*B): row-by-column dot products
- Transpose (A^T): flip rows and columns
- Determinant of 2x2: det([[a,b],[c,d]]) = ad - bc
- Inverse: A^(-1) exists iff det(A) != 0
Eigenvalues and Eigenvectors:
- Av = lambda*v  where lambda = eigenvalue, v = eigenvector
- Find eigenvalues: det(A - lambda*I) = 0 (characteristic equation)
- Eigenvectors: solve (A - lambda*I)v = 0
Vector Spaces:
- Span, basis, dimension
- Linear independence: no vector is a combination of others
- Rank-Nullity Theorem: rank(A) + nullity(A) = n (columns)
Dot Product & Norms:
- u dot v = |u||v|cos(theta)
- ||v|| = sqrt(v1^2 + v2^2 + ... + vn^2)
- Orthogonal: u dot v = 0""",
    metadata={"source": "knowledge_base", "topic": "linear_algebra"}),

    Document(page_content="""Statistics & Probability:
Descriptive Statistics:
- Mean: mu = sum(x) / n
- Variance: sigma^2 = sum((x - mu)^2) / n
- Standard Deviation: sigma = sqrt(variance)
- Median: middle value when sorted
- Mode: most frequent value
Probability Rules:
- P(A union B) = P(A) + P(B) - P(A intersection B)
- P(A | B) = P(A intersection B) / P(B)
- Bayes Theorem: P(A|B) = P(B|A)*P(A) / P(B)
- Independent events: P(A intersection B) = P(A) * P(B)
Distributions:
- Normal: bell curve, described by mu and sigma
- Binomial: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
- Poisson: P(X=k) = (lambda^k * e^(-lambda)) / k!
Central Limit Theorem:
Sample means approach normal distribution as n approaches infinity""",
    metadata={"source": "knowledge_base", "topic": "statistics"}),

    Document(page_content="""Algebra & Number Theory:
Quadratic Formula:
x = (-b +/- sqrt(b^2 - 4ac)) / 2a  for ax^2 + bx + c = 0
Discriminant: D = b^2 - 4ac
- D > 0: two real roots
- D = 0: one real root (repeated)
- D < 0: two complex roots
Logarithm Rules:
- log(ab) = log(a) + log(b)
- log(a/b) = log(a) - log(b)
- log(a^n) = n*log(a)
- log_b(x) = ln(x) / ln(b)
Polynomial Factoring Patterns:
- a^2 - b^2 = (a+b)(a-b)
- a^3 + b^3 = (a+b)(a^2 - ab + b^2)
- a^3 - b^3 = (a-b)(a^2 + ab + b^2)
Sequences & Series:
- Arithmetic: a_n = a_1 + (n-1)d,  Sum = n(a_1 + a_n)/2
- Geometric: a_n = a_1 * r^(n-1),  Sum = a_1(1-r^n)/(1-r)
- Infinite geometric (|r|<1): Sum = a_1 / (1-r)""",
    metadata={"source": "knowledge_base", "topic": "algebra"}),

    Document(page_content="""Trigonometry:
Unit Circle & Basic Identities:
- sin^2(x) + cos^2(x) = 1
- tan(x) = sin(x)/cos(x)
- sec(x) = 1/cos(x),  csc(x) = 1/sin(x),  cot(x) = 1/tan(x)
Angle Sum Formulas:
- sin(A+B) = sin(A)cos(B) + cos(A)sin(B)
- cos(A+B) = cos(A)cos(B) - sin(A)sin(B)
Double Angle:
- sin(2x) = 2sin(x)cos(x)
- cos(2x) = cos^2(x) - sin^2(x)
Key Values:
- sin(0)=0, sin(pi/6)=1/2, sin(pi/4)=sqrt(2)/2, sin(pi/3)=sqrt(3)/2, sin(pi/2)=1
- cos(0)=1, cos(pi/6)=sqrt(3)/2, cos(pi/4)=sqrt(2)/2, cos(pi/3)=1/2, cos(pi/2)=0
Law of Sines: a/sin(A) = b/sin(B) = c/sin(C)
Law of Cosines: c^2 = a^2 + b^2 - 2ab*cos(C)""",
    metadata={"source": "knowledge_base", "topic": "trigonometry"}),

    Document(page_content="""Discrete Mathematics:
Combinatorics:
- Permutations (ordered): P(n,r) = n! / (n-r)!
- Combinations (unordered): C(n,r) = n! / (r!(n-r)!)
- Pigeonhole Principle: n+1 objects in n boxes means at least one box has 2+ objects
Graph Theory:
- Euler path: visits every edge once (exists if 0 or 2 odd-degree vertices)
- Hamiltonian path: visits every vertex once
- Tree: connected graph with n-1 edges for n vertices
- Degree sum = 2 * number of edges
Number Theory:
- GCD(a,b) via Euclidean algorithm: GCD(a,b) = GCD(b, a mod b)
- LCM(a,b) = a*b / GCD(a,b)
- Modular arithmetic: a congruent to b (mod n) means n divides (a-b)
- Fermat's Little Theorem: a^(p-1) congruent to 1 (mod p) for prime p
Logic:
- De Morgan's: NOT(A AND B) = NOT(A) OR NOT(B)
- De Morgan's: NOT(A OR B) = NOT(A) AND NOT(B)""",
    metadata={"source": "knowledge_base", "topic": "discrete_math"}),
]


class MathDataLoader:
    def __init__(self):
        self.documents = []

    def load_builtin_knowledge(self):
        logger.info(f"Loading {len(MATH_KNOWLEDGE_BASE)} built-in knowledge documents")
        return list(MATH_KNOWLEDGE_BASE)

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


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — DATA PREPROCESSING                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 — SPLITTING AND CHUNKING                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — EMBEDDINGS, VECTOR DB & KNOWLEDGE BASE                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True})


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


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 5 — QUERY PROCESSING & AI ENGINE                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

SYSTEM_TEMPLATE = """You are a mathematics teacher solving problems on a whiteboard.

EXACT FORMAT — follow this every time:

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question: [restate the question]
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — [title]
   [working]

Step 2 — [title]
   [working]

━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Answer: [final answer]
━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRICT RULES:
- Each step must be DIFFERENT — never write the same line twice
- Simple problems need only 1 step — do not pad with fake steps
- Each step must add NEW information — if you have nothing new to say, STOP
- Maximum 5 steps total
- Use $...$ for all math
- NEVER repeat a step
- STOP after the answer — do not add notes or commentary

Context:
{context}
"""


class MongoDBChatMemory:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.collection = None
        self._memory: List[Dict] = []
        self._connect()

    def _connect(self):
        if not MONGODB_URI:
            return
        try:
            from pymongo import MongoClient
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            self.collection = client[MONGODB_DB_NAME][MONGODB_COLLECTION]
            logger.info("MongoDB connected")
        except Exception as e:
            logger.warning(f"MongoDB unavailable ({e}), using in-memory history")

    def add_message(self, role: str, content: str):
        msg = {"session_id": self.session_id, "role": role,
               "content": content, "timestamp": datetime.utcnow()}
        if self.collection is not None:
            try:
                self.collection.insert_one(msg)
                return
            except Exception:
                pass
        self._memory.append(msg)

    def get_history(self, limit: int = 20) -> List[Dict]:
        if self.collection is not None:
            try:
                msgs = list(self.collection.find(
                    {"session_id": self.session_id}).sort("timestamp", -1).limit(limit))
                msgs.reverse()
                return msgs
            except Exception:
                pass
        return self._memory[-limit:]

    def get_langchain_messages(self, limit: int = 10):
        history = self.get_history(limit)
        result = []
        for msg in history:
            if msg["role"] == "human":
                result.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                result.append(AIMessage(content=msg["content"]))
        return result

    def clear_history(self):
        if self.collection is not None:
            try:
                self.collection.delete_many({"session_id": self.session_id})
            except Exception:
                pass
        self._memory.clear()


class SymbolicMathEngine:
    @staticmethod
    def differentiate(expression: str, variable: str = "x") -> Optional[str]:
        try:
            import sympy as sp
            var = sp.Symbol(variable)
            expr = sp.sympify(re.sub(r'\^', '**', expression.strip()))
            return f"d/d{variable}[{expression}] = {sp.simplify(sp.diff(expr, var))}"
        except Exception:
            return None

    @staticmethod
    def integrate(expression: str, variable: str = "x") -> Optional[str]:
        try:
            import sympy as sp
            var = sp.Symbol(variable)
            expr = sp.sympify(re.sub(r'\^', '**', expression.strip()))
            return f"integral({expression}) d{variable} = {sp.integrate(expr, var)} + C"
        except Exception:
            return None

    @staticmethod
    def solve_equation(equation: str, variable: str = "x") -> Optional[str]:
        try:
            import sympy as sp
            var = sp.Symbol(variable)
            eq_str = re.sub(r'\^', '**', equation.strip())
            if "=" in eq_str:
                lhs, rhs = eq_str.split("=", 1)
                eq = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
            else:
                eq = sp.sympify(eq_str)
            return f"Solutions for {variable}: {sp.solve(eq, var)}"
        except Exception:
            return None

    @staticmethod
    def try_solve(expression: str) -> Optional[str]:
        try:
            import sympy as sp
            x, y, z, t = sp.symbols('x y z t')
            expr_str = re.sub(r'\^', '**', expression.strip())
            result = sp.simplify(sp.sympify(expr_str, locals={
                'x': x, 'y': y, 'z': z, 't': t,
                'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp,
                'log': sp.log, 'sqrt': sp.sqrt, 'pi': sp.pi}))
            return str(result)
        except Exception:
            return None

    @staticmethod
    def matrix_operations(matrix_str: str) -> Optional[Dict[str, Any]]:
        try:
            import sympy as sp
            M = sp.Matrix(eval(matrix_str))
            return {"determinant": str(M.det()), "rank": M.rank(),
                    "eigenvalues": str(M.eigenvals()), "trace": str(M.trace())}
        except Exception:
            return None


class MathAIEngine:
    def __init__(self, vector_store: MathVectorStore = None, session_id: str = "default"):
        self.llm          = self._init_llm()
        self.vector_store = vector_store
        self.memory       = MongoDBChatMemory(session_id=session_id)
        self.symbolic     = SymbolicMathEngine()
        self.session_id   = session_id

    def _init_llm(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")
        from langchain_groq import ChatGroq
        logger.info(f"Initializing Groq LLM: {LLM_MODEL}")
        return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL,
                        temperature=0.1, max_tokens=2048)

    def _retrieve_context(self, query: str) -> Tuple[list, str]:
        if not self.vector_store or not self.vector_store.is_ready():
            return [], "No knowledge base available. Using general mathematical knowledge."
        docs = self.vector_store.similarity_search(query, k=5)
        if not docs:
            return [], "No specific context found."
        parts = [f"[Reference {i+1} - {d.metadata.get('topic','math')}]\n{d.page_content}"
                 for i, d in enumerate(docs)]
        return docs, "\n\n---\n\n".join(parts)

    def _symbolic_hint(self, query: str) -> Optional[str]:
        ql = query.lower()
        for pattern, action in [
            (r"(?:differentiate|derivative of|d/dx)\s+(.+?)(?:\s+with respect|\s*$)", "diff"),
            (r"(?:integrate|integral of)\s+(.+?)(?:\s+with respect|\s+dx|\s*$)", "int"),
            (r"solve\s+(.+?)\s+(?:for|=)", "solve"),
        ]:
            m = re.search(pattern, ql)
            if m:
                expr = m.group(1).strip()
                result = (self.symbolic.differentiate(expr) if action == "diff"
                          else self.symbolic.integrate(expr) if action == "int"
                          else self.symbolic.solve_equation(expr))
                if result:
                    return f"[Symbolic verification: {result}]"
        return None

    def query(self, user_input: str) -> Dict[str, Any]:
        hint        = self._symbolic_hint(user_input)
        source_docs, context = self._retrieve_context(user_input)
        chat_history = self.memory.get_langchain_messages(limit=10)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        try:
            messages = prompt.format_messages(
                context=context, chat_history=chat_history, input=user_input)
            answer = self.llm.invoke(messages).content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = f"Error: {e}\n\nPlease check your GROQ_API_KEY."

        self.memory.add_message("human", user_input)
        self.memory.add_message("assistant", answer)

        sources = [{"topic":      d.metadata.get("topic", "unknown"),
                    "source":     d.metadata.get("source", "kb"),
                    "difficulty": d.metadata.get("difficulty", "unknown")}
                   for d in source_docs]

        return {"answer": answer, "sources": sources, "symbolic_hint": hint,
                "session_id": self.session_id, "context_docs": len(source_docs)}

    def clear_memory(self):
        self.memory.clear_history()

    def get_history(self):
        return self.memory.get_history(limit=50)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  OCR / IMAGE SCAN HELPER                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

def ocr_extract_text(image) -> str:
    try:
        from PIL import Image
        import PIL.ImageEnhance as enhance
        import pytesseract
        gray      = image.convert("L")
        contrast  = enhance.Contrast(gray).enhance(2.0)
        sharpened = enhance.Sharpness(contrast).enhance(2.0)
        text = pytesseract.image_to_string(sharpened, config='--psm 6').strip()
        text = ' '.join(text.replace('\n', ' ').split())
        return text
    except ImportError:
        return "ERROR_NO_PYTESSERACT"
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 6 — STREAMLIT UI                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

def render_graph(expression: str, x_range: tuple = (-10, 10), title: str = ""):
    import streamlit as st
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#1e2235")
        ax.set_facecolor("#0f1117")
        x = np.linspace(x_range[0], x_range[1], 1000)
        ns = {"__builtins__": {}, "x": x, "np": np,
              "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
              "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
              "pi": np.pi, "e": np.e,
              "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan}
        colors = ["#4f8ef7", "#4ecca3", "#f5c842", "#ff6b6b", "#c792ea"]
        for i, expr in enumerate(expression.split(",")[:5]):
            try:
                y = eval(re.sub(r'\^', '**', expr.strip()), ns)
                y = np.where(np.abs(y) > 1e10, np.nan, y)
                ax.plot(x, y, color=colors[i % len(colors)], linewidth=2.2,
                        label=f"y = {expr.strip()}", alpha=0.9)
            except Exception:
                st.warning(f"Could not plot: {expr.strip()}")
        ax.axhline(0, color="#4a5568", linewidth=0.8, alpha=0.7)
        ax.axvline(0, color="#4a5568", linewidth=0.8, alpha=0.7)
        ax.grid(True, alpha=0.15, color="#4a5568", linestyle="--")
        for spine in ax.spines.values():
            spine.set_color("#2d3561")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors="#8892b0")
        ax.set_xlabel("x", color="#8892b0", fontsize=11)
        ax.set_ylabel("y", color="#8892b0", fontsize=11)
        if title:
            ax.set_title(title, color="#e8eaf6", fontsize=13, pad=15)
        if "," in expression:
            ax.legend(facecolor="#1e2235", edgecolor="#2d3561",
                      labelcolor="#e8eaf6", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Graph error: {e}")


def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Advanced Mathematics Assistant",
                       page_icon="∫", layout="wide")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
    :root{--bg:#0f1117;--bg2:#1a1d2e;--card:#1e2235;--blue:#4f8ef7;--green:#4ecca3;--gold:#f5c842;--tx:#e8eaf6;--tx2:#8892b0;--border:#2d3561;}
    .stApp{background-color:var(--bg);color:var(--tx);}
    .main-h{font-family:'Merriweather',serif;font-size:2.1rem;font-weight:700;color:var(--blue);text-align:center;padding:1.5rem 0 .3rem;}
    .sub-h{font-family:'Inter',sans-serif;font-size:.95rem;color:var(--tx2);text-align:center;margin-bottom:1.5rem;}
    .msg-u{background:linear-gradient(135deg,#1a2a4a,#1e3057);border-left:3px solid var(--blue);padding:1rem 1.2rem;border-radius:0 12px 12px 12px;margin:.6rem 0;font-family:'Inter',sans-serif;font-size:0.95rem;}
    .tag{display:inline-block;background:#1a2a3a;color:var(--gold);font-size:.7rem;font-family:'JetBrains Mono',monospace;padding:2px 8px;border-radius:4px;margin:2px;border:1px solid #2a3a4a;}
    .hint{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--gold);padding:3px 10px;background:#1a1a2e;border-radius:4px;margin:4px 0;display:inline-block;}
    .stButton>button{background:linear-gradient(135deg,var(--blue),#3a6fd8)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:500!important;}
    .stTextArea textarea{font-family:'JetBrains Mono',monospace!important;font-size:.9rem!important;background-color:var(--bg2)!important;color:var(--tx)!important;border:1px solid var(--border)!important;}
    </style>
    """, unsafe_allow_html=True)

    for k, v in [("session_id", str(uuid.uuid4())[:8]), ("messages", []),
                 ("engine", None), ("kb_ready", False),
                 ("query_count", 0), ("pending", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        st.markdown("## ∫ Math Assistant")
        st.markdown(f"<small style='color:#8892b0'>Session `{st.session_state.session_id}`</small>",
                    unsafe_allow_html=True)
        st.divider()

        if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
            st.success("✅ Groq API Connected")
        else:
            st.error("❌ Add GROQ_API_KEY to .env")

        st.divider()
        st.markdown("**📚 Quick Examples**")
        examples = {
            "🔢 Quadratic":   "Solve 2x² + 5x - 3 = 0 step by step",
            "📐 Derivatives": "Find the derivative of f(x) = x³sin(x)",
            "🔗 Integration": "Evaluate the integral of x²e^x dx",
            "🎯 Eigenvalues": "Find eigenvalues of the matrix [[2,1],[1,2]]",
            "📊 Statistics":  "Explain the Central Limit Theorem with an example",
            "📈 Series":      "Does the series sum(1/n²) converge? Find its sum",
            "🔺 Vectors":     "Find the angle between vectors (1,2,3) and (4,5,6)",
            "🧮 Probability": "Explain Bayes theorem with a medical test example",
        }
        for label, question in examples.items():
            if st.button(label, key=f"ex_{label}", use_container_width=True):
                st.session_state.pending = question
                st.rerun()

        st.divider()
        st.markdown("**📉 Graph Plotter**")
        gexpr  = st.text_input("Function(s):", placeholder="x**2, sin(x)")
        grange = st.slider("x range", -20, 20, (-10, 10))
        gtitle = st.text_input("Title:", placeholder="My Graph")
        if st.button("📊 Plot", use_container_width=True) and gexpr:
            render_graph(gexpr, grange, gtitle)

        st.divider()
        st.markdown("**⚡ Symbolic Compute**")
        sym_in  = st.text_input("Expression:", placeholder="x**3 + 2*x")
        sym_act = st.selectbox("Action:", ["Differentiate", "Integrate", "Solve (=0)", "Simplify"])
        if st.button("⚡ Compute", use_container_width=True) and sym_in:
            sym = SymbolicMathEngine()
            result = (sym.differentiate(sym_in)        if sym_act == "Differentiate" else
                      sym.integrate(sym_in)             if sym_act == "Integrate"    else
                      sym.solve_equation(sym_in + "=0") if sym_act == "Solve (=0)"  else
                      sym.try_solve(sym_in))
            st.success(result) if result else st.warning("Could not compute symbolically")

        # ── PDF Upload ────────────────────────────────────────────────
        st.divider()
        st.markdown("**📄 Upload PDF**")
        st.caption("Upload textbook, notes or question paper")
        uploaded_pdf = st.file_uploader(
            "Choose PDF file", type=["pdf"], label_visibility="collapsed")

        if uploaded_pdf is not None:
            tmp_path = f"temp_{uploaded_pdf.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_pdf.read())
            with st.spinner("📖 Reading PDF..."):
                try:
                    pdf_docs = MathDataLoader().load_pdf(tmp_path)
                    clean    = MathDataPreprocessor().preprocess_documents(pdf_docs)
                    chunks   = MathTextSplitter().split_documents(clean)
                    if st.session_state.engine and chunks:
                        st.session_state.engine.vector_store.add_documents(chunks)
                        st.success(f"✅ Loaded {len(pdf_docs)} page(s) from PDF!")
                        # ── NEW: PDF action buttons ───────────────────
                        st.markdown("**Ask about this PDF:**")
                        if st.button("📝 Solve all problems in this PDF", use_container_width=True, type="primary"):
                            st.session_state.pending = "Solve all the math problems from the uploaded PDF step by step"
                            st.rerun()
                        if st.button("📋 Summarize this PDF", use_container_width=True):
                            st.session_state.pending = "Summarize the key math concepts from the uploaded PDF"
                            st.rerun()
                        if st.button("❓ What topics are in this PDF?", use_container_width=True):
                            st.session_state.pending = "What math topics are covered in the uploaded PDF?"
                            st.rerun()
                    else:
                        st.warning("⚠️ No content found in PDF")
                except Exception as e:
                    st.error(f"PDF error: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        # ── Camera / Image Scan ───────────────────────────────────────
        st.divider()
        st.markdown("**📷 Scan Math Problem**")
        st.caption("Snap or upload a photo of any math problem")

        scan_method = st.radio(
            "Choose input:",
            ["📷 Use Camera", "🖼️ Upload Image"],
            horizontal=True,
            label_visibility="collapsed")

        scanned_image = None
        if scan_method == "📷 Use Camera":
            scanned_image = st.camera_input(
                "Point at math problem and capture", label_visibility="collapsed")
        else:
            scanned_image = st.file_uploader(
                "Upload photo of math problem",
                type=["png", "jpg", "jpeg", "webp"],
                label_visibility="collapsed",
                key="img_uploader")

        if scanned_image is not None:
            try:
                from PIL import Image as PILImage
                image = PILImage.open(scanned_image)
                st.image(image, caption="📸 Captured Image", use_column_width=True)

                with st.spinner("🔍 Reading math problem from image..."):
                    extracted = ocr_extract_text(image)

                if extracted == "ERROR_NO_PYTESSERACT":
                    st.error("❌ pytesseract not installed!")
                    st.code("pip install pytesseract pillow\nbrew install tesseract")

                elif extracted:
                    st.success("✅ Problem detected!")
                    st.info(f"📝 **Detected text:** {extracted}")

                    # ── Edit box — in case OCR made a mistake ─────────
                    st.markdown("<small style='color:#8892b0'>✏️ Edit if OCR made a mistake, then choose action:</small>",
                                unsafe_allow_html=True)
                    edited = st.text_input(
                        "Edit detected text:", value=extracted, label_visibility="collapsed")

                    # ── Action buttons ────────────────────────────────
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("🧮 Solve", use_container_width=True, type="primary"):
                            st.session_state.pending = f"Solve this math problem step by step: {edited}"
                            st.rerun()
                        if st.button("📋 Summarize", use_container_width=True):
                            st.session_state.pending = f"Summarize and explain this math problem: {edited}"
                            st.rerun()
                    with b2:
                        if st.button("💡 Give Hint", use_container_width=True):
                            st.session_state.pending = f"Give me a hint to solve: {edited}"
                            st.rerun()
                        if st.button("📊 Similar Examples", use_container_width=True):
                            st.session_state.pending = f"Show me similar example problems like: {edited}"
                            st.rerun()

                    # ── AUTO SOLVE — fires immediately after OCR ──────
                    if "last_scanned" not in st.session_state or st.session_state.last_scanned != extracted:
                        st.session_state.last_scanned = extracted
                        st.session_state.pending = f"Solve this math problem step by step: {extracted}"
                        st.rerun()

                else:
                    st.warning("⚠️ Could not read text. Try better lighting or type manually.")
                    st.info("💡 Tips: Good lighting, flat surface, clear handwriting works best!")

            except ImportError:
                st.error("❌ Pillow not installed. Run: pip install pillow")
            except Exception as e:
                st.error(f"Scan error: {e}")

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Queries",  st.session_state.query_count)
        c2.metric("Messages", len(st.session_state.messages))
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages    = []
            st.session_state.query_count = 0
            if st.session_state.engine:
                st.session_state.engine.clear_memory()
            st.rerun()

    st.markdown('<h1 class="main-h">Advanced Mathematics Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-h">Powered by LLaMA 3 × Groq × RAG Knowledge Base</p>', unsafe_allow_html=True)

    if st.session_state.engine is None:
        with st.spinner("🔧 Building knowledge base (first run only)..."):
            try:
                store = build_pipeline()
                st.session_state.engine   = MathAIEngine(
                    vector_store=store, session_id=st.session_state.session_id)
                st.session_state.kb_ready = True
            except Exception as e:
                st.error(f"Init error: {e}")
                st.info("Make sure GROQ_API_KEY is set in .env and all packages are installed.")
                st.stop()

    if st.session_state.kb_ready and st.session_state.engine:
        doc_count = (st.session_state.engine.vector_store.get_document_count()
                     if st.session_state.engine.vector_store else 0)
        st.caption(f"📚 {doc_count} chunks indexed | Model: {LLM_MODEL} | Session: {st.session_state.session_id}")

    st.divider()

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;color:#8892b0">
            <div style="font-size:3rem;margin-bottom:1rem">∫ ∑ ∂ π</div>
            <h3 style="color:#e8eaf6">Welcome to the Math Assistant</h3>
            <p>Ask me anything — algebra, calculus, linear algebra, statistics, and beyond.</p>
            <p>📷 <strong>New:</strong> Scan handwritten problems with your camera!</p>
            <p>📄 <strong>New:</strong> Upload PDF textbooks or question papers!</p>
            <p style="font-size:.85rem">Use the sidebar for graphs, symbolic compute, PDF upload and camera scan.</p>
        </div>""", unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-u"><strong>🧑 You</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="border-left:3px solid #4ecca3;padding-left:10px;'
                'color:#4ecca3;font-size:0.85rem;font-weight:700;margin:16px 0 6px 0;">'
                '∫ Assistant</p>',
                unsafe_allow_html=True)
            st.markdown(msg["content"])
            st.divider()
            clean = re.sub(r'\$\$(.+?)\$\$', r'\1', msg["content"], flags=re.DOTALL)
            clean = re.sub(r'\$(.+?)\$', r'\1', clean)
            clean = re.sub(r'━+', '─────────────────', clean)
            with st.expander("📋 Copy plain text", expanded=False):
                st.code(clean, language=None)
            if msg.get("sources"):
                tags = "".join(
                    f'<span class="tag">📖 {s["topic"].replace("_", " ").title()}</span>'
                    for s in msg["sources"])
                st.markdown(f'<div style="margin-bottom:8px">{tags}</div>',
                            unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns([5, 1])
    with col1:
        default    = st.session_state.get("pending") or ""
        user_input = st.text_area(
            "Question:", value=default, height=100,
            placeholder="e.g., Solve x² - 5x + 6 = 0   or   Find the derivative of sin(x)·x²",
            key="user_input", label_visibility="collapsed")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        send = st.button("Ask ∫", use_container_width=True, type="primary")

    if st.session_state.get("pending"):
        st.session_state.pending = None

    if send and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        st.session_state.query_count += 1
        with st.spinner("🧮 Computing solution..."):
            result = st.session_state.engine.query(user_input.strip())
        st.session_state.messages.append({
            "role":          "assistant",
            "content":       result["answer"],
            "sources":       result.get("sources", []),
            "symbolic_hint": result.get("symbolic_hint"),
        })
        st.rerun()

    with st.expander("💡 Tips", expanded=False):
        st.markdown("""
        - **Type question** → click Ask ∫ for step-by-step solution
        - **📷 Camera** → snap photo of handwritten problem → auto solves!
        - **🖼️ Upload Image** → upload screenshot or photo of any problem
        - **📄 PDF Upload** → upload textbook or notes → 3 action buttons appear!
        - **Graph**: Type `x**2, sin(x)` in sidebar Graph Plotter
        - **Symbolic**: Use ⚡ Compute for instant derivatives/integrals
        - **Copy**: Click 📋 Copy plain text below any answer
        """)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 7 — TESTING                                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

class TestDataSources(unittest.TestCase):
    def test_builtin_knowledge(self):
        docs = MathDataLoader().load_builtin_knowledge()
        self.assertGreater(len(docs), 0)
        self.assertTrue(all(len(d.page_content) > 50 for d in docs))
        self.assertTrue(all("topic" in d.metadata for d in docs))

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.pp = MathDataPreprocessor()

    def test_cleans_whitespace(self):
        doc = Document(page_content="Hello   World\n\n\n\nMath content here is important", metadata={})
        result = self.pp.preprocess_document(doc)
        self.assertIsNotNone(result)
        self.assertNotIn("\n\n\n", result.page_content)

    def test_deduplication(self):
        doc = Document(page_content="The derivative of x squared is 2x. " * 10, metadata={})
        self.assertIsNotNone(self.pp.preprocess_document(doc))
        self.assertIsNone(self.pp.preprocess_document(doc))

    def test_topic_detection(self):
        doc = Document(page_content="The derivative and integral of functions in calculus.", metadata={})
        result = self.pp.preprocess_document(doc)
        self.assertEqual(result.metadata["topic"], "calculus")

    def test_skips_short(self):
        self.assertIsNone(MathDataPreprocessor().preprocess_document(
            Document(page_content="too short", metadata={})))

class TestChunking(unittest.TestCase):
    def setUp(self):
        self.splitter = MathTextSplitter(chunk_size=200, chunk_overlap=20)

    def test_splits_large_doc(self):
        doc = Document(page_content="Math paragraph with content. " * 60, metadata={"source": "test"})
        self.assertGreater(len(self.splitter.split_document(doc)), 1)

    def test_metadata_preserved(self):
        doc = Document(page_content="x " * 500, metadata={"source": "test.pdf", "topic": "algebra"})
        for chunk in self.splitter.split_document(doc):
            self.assertEqual(chunk.metadata.get("topic"), "algebra")
            self.assertIn("chunk_index", chunk.metadata)

class TestSymbolicEngine(unittest.TestCase):
    def setUp(self):
        self.sym = SymbolicMathEngine()

    def test_differentiate(self):
        result = self.sym.differentiate("x**3")
        self.assertIsNotNone(result)
        self.assertIn("3*x**2", result.replace(" ", ""))

    def test_integrate(self):
        result = self.sym.integrate("x**2")
        self.assertIsNotNone(result)
        self.assertIn("x**3", result)

    def test_solve(self):
        result = self.sym.solve_equation("x**2 - 4 = 0")
        self.assertIsNotNone(result)
        self.assertIn("2", result)

    def test_simplify(self):
        result = self.sym.try_solve("(x**2 - 1)/(x - 1)")
        self.assertIsNotNone(result)
        self.assertIn("x + 1", result)

class TestMemory(unittest.TestCase):
    def test_add_retrieve(self):
        mem = MongoDBChatMemory(session_id="test")
        mem.add_message("human", "What is a derivative?")
        mem.add_message("assistant", "Rate of change.")
        self.assertGreaterEqual(len(mem.get_history()), 2)

    def test_langchain_messages(self):
        mem = MongoDBChatMemory(session_id="test_lc")
        mem.add_message("human", "Test")
        mem.add_message("assistant", "Answer")
        msgs = mem.get_langchain_messages()
        self.assertTrue(any(isinstance(m, HumanMessage) for m in msgs))


def run_tests() -> bool:
    print("\n" + "="*60 + "\n  RUNNING TEST SUITE\n" + "="*60)
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestDataSources, TestPreprocessing, TestChunking, TestSymbolicEngine, TestMemory]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def run_evaluation():
    print("\n" + "="*60 + "\n  RAG PIPELINE EVALUATION\n" + "="*60)
    store = build_pipeline()
    test_cases = [
        {"q": "What is the power rule for derivatives?",  "keywords": ["power", "derivative", "n*x"]},
        {"q": "How do you find eigenvalues of a matrix?", "keywords": ["eigenvalue", "determinant", "characteristic"]},
        {"q": "What is Bayes theorem?",                   "keywords": ["probability", "conditional", "P(A|B)"]},
        {"q": "What is the quadratic formula?",           "keywords": ["quadratic", "formula", "discriminant"]},
    ]
    scores = []
    for tc in test_cases:
        docs     = store.similarity_search(tc["q"], k=3)
        combined = " ".join(d.page_content.lower() for d in docs)
        hits     = sum(1 for kw in tc["keywords"] if kw.lower() in combined)
        score    = hits / len(tc["keywords"])
        scores.append(score)
        print(f"  Q: {tc['q'][:50]}... → {score:.2f} ({hits}/{len(tc['keywords'])} keywords)")
    print(f"\n  Average retrieval score: {sum(scores)/len(scores):.3f}")
    sym      = SymbolicMathEngine()
    sym_tests = [
        (sym.differentiate("x**3"),           "3*x**2"),
        (sym.integrate("x**2"),               "x**3"),
        (sym.solve_equation("x**2 - 4 = 0"),  "2"),
    ]
    passed = sum(1 for r, e in sym_tests if r and e in r.replace(" ", ""))
    print(f"  Symbolic engine: {passed}/{len(sym_tests)} tests passed")


def main():
    if any("streamlit" in arg for arg in sys.argv):
        run_streamlit_app()
        return

    parser = argparse.ArgumentParser(description="Advanced Mathematics Assistant")
    parser.add_argument("--setup",   action="store_true", help="Build knowledge base")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild knowledge base")
    parser.add_argument("--test",    action="store_true", help="Run unit tests")
    parser.add_argument("--eval",    action="store_true", help="Evaluate RAG pipeline")
    args = parser.parse_args()

    print("="*60 + "\n  🧮  Advanced Mathematics Assistant\n" + "="*60)

    if args.test:
        sys.exit(0 if run_tests() else 1)
    elif args.eval:
        run_evaluation()
    elif args.setup or args.rebuild:
        store = build_pipeline(force_rebuild=args.rebuild)
        print(f"\n✅ Knowledge base ready — {store.get_document_count()} chunks indexed")
    else:
        print("\nTo launch the UI:\n")
        print("  python3.11 -m streamlit run main.py\n")


try:
    import streamlit as st
    if hasattr(st, "session_state"):
        run_streamlit_app()
except Exception:
    pass

if __name__ == "__main__":
    main()