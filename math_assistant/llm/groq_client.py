import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import JsonOutputParser
except ImportError:
    from langchain.schema import HumanMessage, AIMessage
    from langchain.schema.output_parser import JsonOutputParser

try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

from math_assistant.rag.vector_store import MathVectorStore

logger = logging.getLogger("math_assistant")

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
LLM_MODEL          = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MONGODB_URI        = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME    = os.getenv("MONGODB_DB_NAME", "math_assistant")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chat_history")

SYSTEM_TEMPLATE = """You are a precision mathematics teacher. 
CRITICAL RULE: You must solve the EXACT problem the user provides. 
- NEVER change the numbers. 
- NEVER change the wording (e.g., if asked for rational, do not prove irrational).
- If the problem is trivial (like root 1), solve it exactly as stated (root 1 = 1, which is rational). Do NOT turn it into a classic proof (like root 2).

EXACT FORMAT — follow this every time:

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question: [restate the EXACT question]
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — [title]
   [working]

Step 2 — [title]
   [working]

━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Answer: [final answer]
━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRICT RULES:
- Use $...$ for all math.
- Simple problems need only 1 step — do not pad.
- STOP after the answer — no commentary.

{sympy_context}

Context:
{context}
"""

INTENT_CLASSIFICATION_TEMPLATE = """You are a mathematical intent classifier. 
Determine if the user's input asks to computationally solve a mathematical problem (e.g., equations, differentiation, integration, matrix operations, limits, taylor series, differential equations) OR if it is a conceptual/theoretical question.

Here is the recent conversation history to help you understand phrases like "integrate it" or "what is the limit of that":
{history}

Respond ONLY with a valid JSON strictly following this schema:
{{
    "is_computation": boolean, 
    "operation": "differentiate" | "integrate" | "solve_equation" | "simplify" | "matrix" | "limit" | "taylor" | "differential_equation" | "none",
    "expression": "string with the exact mathematical expression to process, use ** for exponents, or null if none",
    "variable": "string with the variable to solve/differentiate/integrate/limit for (usually x), or null if none",
    "point": "string representing the point to evaluate a limit or taylor series at (e.g., '0', 'oo' for infinity), or null",
    "degree": "integer representing the degree for a taylor series expansion, or null"
}}

User input: {input}
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
            return f"{sp.simplify(sp.diff(expr, var))}"
        except Exception:
            return None

    @staticmethod
    def integrate(expression: str, variable: str = "x") -> Optional[str]:
        try:
            import sympy as sp
            var = sp.Symbol(variable)
            expr = sp.sympify(re.sub(r'\^', '**', expression.strip()))
            return f"{sp.integrate(expr, var)}"
        except Exception:
            return None

    @staticmethod
    def evaluate_limit(expression: str, variable: str = "x", point: str = "0") -> Optional[str]:
        try:
            import sympy as sp
            var = sp.Symbol(variable)
            pt = sp.sympify(point)
            expr = sp.sympify(re.sub(r'\^', '**', expression.strip()))
            return f"Limit of {expression} as {variable} -> {point} is {sp.limit(expr, var, pt)}"
        except Exception:
            return None

    @staticmethod
    def taylor_series(expression: str, variable: str = "x", point: str = "0", degree: int = 5) -> Optional[str]:
        try:
            import sympy as sp
            var = sp.Symbol(variable)
            pt = sp.sympify(point)
            expr = sp.sympify(re.sub(r'\^', '**', expression.strip()))
            return f"Taylor series of {expression} around {variable}={point} (degree {degree}): {sp.series(expr, var, pt, degree).removeO()}"
        except Exception:
            return None

    @staticmethod
    def solve_differential_equation(equation: str) -> Optional[str]:
        try:
            import sympy as sp
            x = sp.Symbol('x')
            f = sp.Function('f')(x)
            
            eq_str = re.sub(r'\^', '**', equation.strip())
            eq_str = eq_str.replace("y''", "Derivative(f, x, x)")
            eq_str = eq_str.replace("y'", "Derivative(f, x)")
            eq_str = eq_str.replace("y", "f")
            
            if "=" in eq_str:
                lhs, rhs = eq_str.split("=", 1)
                eq = sp.Eq(sp.sympify(lhs, locals={'f': f, 'x': x, 'Derivative': sp.Derivative}), 
                           sp.sympify(rhs, locals={'f': f, 'x': x, 'Derivative': sp.Derivative}))
            else:
                eq = sp.sympify(eq_str, locals={'f': f, 'x': x, 'Derivative': sp.Derivative})
            
            sol = sp.dsolve(eq, f)
            return f"General Solution: {sol.rhs}"
        except Exception as e:
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

    def _classify_and_extract(self, user_input: str) -> Dict[str, Any]:
        try:
            from langchain_core.output_parsers import JsonOutputParser
            parser = JsonOutputParser()
            prompt = PromptTemplate(
                template=INTENT_CLASSIFICATION_TEMPLATE,
                input_variables=["history", "input"]
            )
            chain = prompt | self.llm | parser
            
            history_list = self.memory.get_history(limit=4)
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_list])
            if not history_text:
                history_text = "No prior history."

            return chain.invoke({"history": history_text, "input": user_input})
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return {"is_computation": False, "operation": "none", "expression": None, "variable": None}

    def vision_extract_math(self, image_bytes_base64: str) -> str:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage
            
            vision_llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.2-11b-vision-preview", temperature=0.1, max_tokens=1024)
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Extract all mathematical equations, expressions, or text in this image. Format the mathematical parts in raw LaTeX. Do not include any conversational padding, ONLY return the extracted text and formulas."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_bytes_base64}"}}
                ]
            )
            response = vision_llm.invoke([message])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Vision AI failed: {e}")
            return f"Error extracting math from image: {e}"

    def query(self, user_input: str) -> Dict[str, Any]:
        # 1. Classify intent
        intent = self._classify_and_extract(user_input)
        hint = None
        sympy_context = ""
        source_docs = []
        context = ""
        graph_expr = None

        # 2. Mathematical computation path
        if intent.get("is_computation") and intent.get("expression"):
            expr = intent["expression"]
            var  = intent.get("variable") or "x"
            op   = intent.get("operation")
            pt   = intent.get("point") or "0"
            deg  = intent.get("degree") or 5
            
            logger.info(f"SymPy route chosen: {op} on {expr}")
            
            if op == "differentiate":
                hint = self.symbolic.differentiate(expr, var)
                graph_expr = hint
            elif op == "integrate":
                hint = self.symbolic.integrate(expr, var)
                graph_expr = hint
            elif op == "limit":
                hint = self.symbolic.evaluate_limit(expr, var, pt)
            elif op == "taylor":
                hint = self.symbolic.taylor_series(expr, var, pt, int(deg))
            elif op == "differential_equation":
                hint = self.symbolic.solve_differential_equation(expr)
            elif op == "solve_equation":
                hint = self.symbolic.solve_equation(expr, var)
            elif op == "simplify":
                hint = self.symbolic.try_solve(expr)
                graph_expr = hint
            elif op == "matrix":
                res = self.symbolic.matrix_operations(expr)
                if res:
                    hint = f"Matrix operations:\nDeterminant: {res['determinant']}\nRank: {res['rank']}\nEigenvalues: {res['eigenvalues']}\nTrace: {res['trace']}"

            if hint:
                sympy_context = f"CRITICAL REQUIREMENT: A symbolic compute engine has already verified the exact mathematical result for this problem. You MUST state this exact result and work backward to explain it step-by-step.\n\n[VERIFIED SYMPY RESULT]:\n{hint}\n\nDo NOT disagree with this result."

        # 3. Concept/RAG fallback path
        if not sympy_context:
            source_docs, context = self._retrieve_context(user_input)
            
        chat_history = self.memory.get_langchain_messages(limit=10)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        try:
            messages = prompt.format_messages(
                context=context, sympy_context=sympy_context, chat_history=chat_history, input=user_input)
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
                "session_id": self.session_id, "context_docs": len(source_docs), "graph_expr": graph_expr}

    def clear_memory(self):
        self.memory.clear_history()

    def get_history(self):
        return self.memory.get_history(limit=50)
