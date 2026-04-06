import unittest
from langchain_core.documents import Document

from math_assistant.rag.retriever import MathDataLoader, MathDataPreprocessor, MathTextSplitter
from math_assistant.llm.groq_client import SymbolicMathEngine, MongoDBChatMemory
from math_assistant.knowledge.build_kb import build_pipeline

try:
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError:
    from langchain.schema import HumanMessage, AIMessage

class TestDataSources(unittest.TestCase):
    def test_builtin_knowledge(self):
        docs = MathDataLoader().load_builtin_knowledge()
        self.assertGreater(len(docs), 0)
        self.assertTrue(all(len(d.page_content) > 50 for d in docs))
        self.assertTrue(all("topic" in d.metadata for d in docs))


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.pp = MathDataPreprocessor()
        self.pp._seen_hashes.clear()


    def test_cleans_whitespace(self):
        doc = Document(page_content="Hello   World\n\n\n\nMath content here is important. " * 3, metadata={})
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
