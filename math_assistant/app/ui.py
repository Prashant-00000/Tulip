import os
import re
import uuid
import logging
from datetime import datetime
import streamlit as st

from math_assistant.llm.groq_client import GROQ_API_KEY, LLM_MODEL, MathAIEngine, SymbolicMathEngine
from math_assistant.rag.retriever import MathDataLoader, MathDataPreprocessor, MathTextSplitter
from math_assistant.knowledge.build_kb import build_pipeline

logger = logging.getLogger("math_assistant")

def ocr_extract_text(image) -> str:
    try:
        import base64
        from io import BytesIO
        if not st.session_state.get('engine'):
            return "Engine not initialized yet. Wait for setup."
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return st.session_state.engine.vision_extract_math(img_str)
    except Exception as e:
        logger.error(f"Vision extraction error: {e}")
        return f"Error: {e}"


def render_graph(expression: str, x_range: tuple = (-10, 10), title: str = ""):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0a0a1a")
        ax.set_facecolor("#0d0d2b")
        x = np.linspace(x_range[0], x_range[1], 1000)
        ns = {"__builtins__": {}, "x": x, "np": np,
              "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
              "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
              "pi": np.pi, "e": np.e,
              "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan}
        colors = ["#a78bfa", "#06d6a0", "#fbbf24", "#f472b6", "#7c3aed"]
        for i, expr in enumerate(expression.split(",")[:5]):
            try:
                y = eval(re.sub(r'\^', '**', expr.strip()), ns)
                y = np.where(np.abs(y) > 1e10, np.nan, y)
                ax.plot(x, y, color=colors[i % len(colors)], linewidth=2.2,
                        label=f"y = {expr.strip()}", alpha=0.9)
            except Exception:
                st.warning(f"Could not plot: {expr.strip()}")
        ax.axhline(0, color="#3d3d6b", linewidth=0.8, alpha=0.7)
        ax.axvline(0, color="#3d3d6b", linewidth=0.8, alpha=0.7)
        ax.grid(True, alpha=0.12, color="#3d3d6b", linestyle="--")
        for spine in ax.spines.values():
            spine.set_color("#2a2a55")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors="#9d9dcc")
        ax.set_xlabel("x", color="#9d9dcc", fontsize=11)
        ax.set_ylabel("y", color="#9d9dcc", fontsize=11)
        if title:
            ax.set_title(title, color="#f0f0ff", fontsize=13, pad=15)
        if "," in expression:
            ax.legend(facecolor="#12122a", edgecolor="#2a2a55",
                      labelcolor="#f0f0ff", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Graph error: {e}")


def run_streamlit_app():
    st.set_page_config(page_title="Advanced Mathematics Assistant",
                       page_icon="∫", layout="wide")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #09090b;
        --bg-grad: linear-gradient(145deg, #09090b, #18181b);
        --bg2: #18181b;
        --card: rgba(24, 24, 27, 0.6);
        --card-solid: #27272a;
        --accent-primary: #3b82f6; 
        --accent-glow: rgba(59, 130, 246, 0.3);
        --accent-gradient: linear-gradient(135deg, #3b82f6, #8b5cf6);
        --teal: #14b8a6;
        --emerald: #10b981;
        --amber: #f59e0b;
        --tx: #fafafa;
        --tx2: #a1a1aa;
        --border: rgba(255, 255, 255, 0.08);
        --border-hover: rgba(255, 255, 255, 0.15);
        --radius: 12px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Global ─────────────────────────────────── */
    .stApp {
        background: var(--bg-grad) !important;
        background-attachment: fixed !important;
        color: var(--tx) !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* ── Scrollbar ──────────────────────────────── */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--card-solid); border-radius: 4px; border: 2px solid var(--bg); }
    ::-webkit-scrollbar-thumb:hover { background: #3f3f46; }

    /* ── Header ─────────────────────────────────── */
    .main-h {
        font-family: 'Outfit', sans-serif;
        font-size: 2.75rem;
        font-weight: 600;
        text-align: center;
        padding: 2.5rem 0 0.5rem;
        letter-spacing: -0.03em;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 3s ease-in-out infinite alternate;
    }
    .sub-h {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        color: var(--tx2);
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: 0.02em;
        font-weight: 300;
    }

    @keyframes glow {
        0% { text-shadow: 0 0 20px rgba(59, 130, 246, 0.1); }
        100% { text-shadow: 0 0 30px rgba(139, 92, 246, 0.2); }
    }

    /* ── Chat Bubbles ──────────────────────────── */
    .msg-u {
        background: var(--card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        padding: 1.4rem 1.6rem;
        border-radius: var(--radius);
        border-bottom-right-radius: 4px;
        margin: 1.2rem 0;
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        color: var(--tx);
        line-height: 1.6;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.4s ease-out forwards;
    }
    .assistant-label {
        color: var(--accent-primary);
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        margin: 28px 0 10px 0;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .assistant-label::before {
        content: '';
        display: inline-block;
        width: 8px;
        height: 8px;
        background: var(--accent-primary);
        border-radius: 50%;
        box-shadow: 0 0 10px var(--accent-glow);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Source Tags ────────────────────────────── */
    .tag {
        display: inline-flex;
        align-items: center;
        background: rgba(39, 39, 42, 0.7);
        color: var(--tx2);
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        padding: 6px 12px;
        border-radius: 20px;
        margin: 4px;
        border: 1px solid var(--border);
        transition: var(--transition);
        backdrop-filter: blur(4px);
    }
    .tag:hover {
        background: var(--card-solid);
        border-color: var(--tx2);
        color: var(--tx);
        transform: translateY(-1px);
    }
    .hint {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: var(--accent-primary);
        padding: 8px 14px;
        background: rgba(59, 130, 246, 0.05);
        border-left: 3px solid var(--accent-primary);
        border-radius: 0 var(--radius) var(--radius) 0;
        margin: 10px 0;
        display: inline-block;
    }

    /* ── Buttons ────────────────────────────────── */
    .stButton > button {
        background: var(--card) !important;
        color: var(--tx) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        font-weight: 500 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: var(--transition) !important;
        padding: 0.6rem 1.2rem !important;
        backdrop-filter: blur(8px) !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }
    .stButton > button:hover {
        background: var(--card-solid) !important;
        border-color: var(--accent-primary) !important;
        color: #fff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    .stButton > button:active {
        transform: translateY(1px) !important;
    }
    
    /* Primary Action Buttons */
    .stButton > button[kind="primary"] {
        background: var(--accent-gradient) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px var(--accent-glow) !important;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.9 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px var(--accent-glow) !important;
    }

    /* ── Text Area & Inputs ─────────────────────── */
    .stTextArea textarea, .stTextInput input {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem !important;
        background-color: var(--bg2) !important;
        color: var(--tx) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem !important;
        transition: var(--transition) !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent-primary) !important;
        outline: none !important;
        box-shadow: 0 0 0 2px var(--accent-glow), inset 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    /* ── Sidebar ────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(9, 9, 11, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-family: 'Outfit', sans-serif !important;
        color: var(--tx) !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
        letter-spacing: -0.02em !important;
    }

    /* ── Expander ───────────────────────────────── */
    .streamlit-expanderHeader {
        background-color: rgba(24, 24, 27, 0.4) !important;
        border-radius: var(--radius) !important;
        font-family: 'Outfit', sans-serif !important;
        border: 1px solid var(--border) !important;
        color: var(--tx2) !important;
        backdrop-filter: blur(8px) !important;
        transition: var(--transition) !important;
    }
    .streamlit-expanderHeader:hover {
        background-color: rgba(24, 24, 27, 0.8) !important;
        border-color: var(--border-hover) !important;
    }

    /* ── Metrics ────────────────────────────────── */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, var(--card), rgba(39, 39, 42, 0.4)) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 16px 20px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        transition: var(--transition) !important;
    }
    [data-testid="metric-container"]:hover {
        border-color: var(--border-hover) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--tx) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        background: var(--accent-gradient) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--tx2) !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 500 !important;
    }

    /* ── Divider ────────────────────────────────── */
    hr {
        border-color: var(--border) !important;
        margin: 2.5em 0 !important;
        border-style: dashed !important;
    }

    /* ── Selectbox ──────────────────────────────── */
    .stSelectbox > div > div {
        background-color: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        transition: var(--transition) !important;
    }
    .stSelectbox > div > div:hover {
        border-color: var(--border-hover) !important;
    }

    /* ── Success/Error/Warning boxes ────────────── */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: var(--radius) !important;
        font-family: 'Outfit', sans-serif !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    }
    .stSuccess { background: rgba(16, 185, 129, 0.05) !important; border-left: 4px solid var(--emerald) !important; color: var(--tx) !important; }
    .stError { background: rgba(239, 68, 68, 0.05) !important; border-left: 4px solid #ef4444 !important; color: var(--tx) !important; }
    .stWarning { background: rgba(245, 158, 11, 0.05) !important; border-left: 4px solid var(--amber) !important; color: var(--tx) !important; }
    .stInfo { background: rgba(59, 130, 246, 0.05) !important; border-left: 4px solid var(--accent-primary) !important; color: var(--tx) !important; }

    /* ── Welcome Panel ──────────────────────────── */
    .welcome-symbols {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 300;
        letter-spacing: 0.3em;
        opacity: 0.9;
    }
    .welcome-box {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(145deg, rgba(24, 24, 27, 0.7), rgba(9, 9, 11, 0.8));
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
        animation: fadeIn 0.8s ease-out forwards;
        max-width: 800px;
        margin: 0 auto;
    }
    .welcome-box h3 {
        color: var(--tx);
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.8rem;
        margin-bottom: 1.2rem;
        letter-spacing: -0.02em;
    }
    .welcome-box p {
        color: var(--tx2);
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        line-height: 1.7;
        margin-bottom: 0.8rem;
    }
    .welcome-box strong {
        color: var(--tx);
        font-weight: 500;
        background: rgba(255, 255, 255, 0.05);
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

    for k, v in [("session_id", str(uuid.uuid4())[:8]), ("messages", []),
                 ("engine", None), ("kb_ready", False),
                 ("query_count", 0), ("pending", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        st.markdown("## ✦ Math Assistant")
        st.markdown(f"<small style='color:#9d9dcc;font-family:JetBrains Mono,monospace;font-size:0.7rem'>Session <code>{st.session_state.session_id}</code></small>",
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

        # ── Math Topics Explorer ──────────────────────────────────────
        st.divider()
        st.markdown("**🧭 Math Topics Explorer**")
        st.caption("Select a topic to get a comprehensive explanation")
        
        topics_dict = {
            "Calculus": ["Derivatives", "Integration", "Limits", "Taylor Series"],
            "Linear Algebra": ["Matrices", "Eigenvalues & Eigenvectors", "Vector Spaces"],
            "Statistics": ["Probability Distributions", "Hypothesis Testing", "Bayes Theorem"],
            "Algebra": ["Polynomials", "Logarithms", "Sequences & Series"],
            "Trigonometry": ["Unit Circle", "Trig Identities", "Law of Sines/Cosines"]
        }
        
        selected_category = st.selectbox("Category:", list(topics_dict.keys()))
        selected_topic = st.selectbox("Topic:", topics_dict[selected_category])
        
        if st.button("📖 Explain Topic", use_container_width=True, type="primary"):
            st.session_state.pending = f"Please provide a comprehensive and detailed explanation of {selected_topic} in {selected_category}, including key concepts, formulas, and a short example. Format the explanation beautifully."
            st.rerun()

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
                        st.toast(f"✅ Loaded {len(pdf_docs)} page(s) from PDF!", icon="✅")
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
                        st.toast("⚠️ No content found in PDF", icon="⚠️")
                except Exception as e:
                    st.toast(f"PDF error: {e}", icon="❌")
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
                    st.toast("❌ pytesseract not installed!", icon="❌")
                    st.code("pip install pytesseract pillow\nbrew install tesseract")

                elif extracted:
                    st.toast("✅ Problem detected!", icon="✅")
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
                    st.toast("⚠️ Could not read text. Try better lighting or type manually.", icon="⚠️")
                    st.info("💡 Tips: Good lighting, flat surface, clear handwriting works best!")

            except ImportError:
                st.error("❌ Pillow not installed. Run: pip install pillow")
            except Exception as e:
                st.error(f"Scan error: {e}")

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Queries",  st.session_state.query_count)
        c2.metric("Messages", len(st.session_state.messages))
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.messages:
            chat_export = ""
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_export += f"### {role}\n{msg['content']}\n\n---\n\n"
            st.download_button(
                label="📥 Export Chat",
                data=chat_export,
                file_name=f"math_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages    = []
            st.session_state.query_count = 0
            if st.session_state.engine:
                st.session_state.engine.clear_memory()
            st.rerun()

    st.markdown('<h1 class="main-h">Advanced Mathematics Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-h">✦ Powered by LLaMA 3 · Groq · RAG Knowledge Base ✦</p>', unsafe_allow_html=True)

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
        <div class="welcome-box">
            <div class="welcome-symbols">∫ ∑ ∂ π</div>
            <h3>Welcome to the Math Assistant</h3>
            <p>Ask me anything — algebra, calculus, linear algebra, statistics, and beyond.</p>
            <p>📷 <strong>New:</strong> Scan handwritten problems with your camera!</p>
            <p>📄 <strong>New:</strong> Upload PDF textbooks or question papers!</p>
            <p style="font-size:.85rem;margin-top:1rem">Use the sidebar for graphs, symbolic compute, PDF upload and camera scan.</p>
        </div>""", unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":

                if msg.get("sources"):
                    with st.expander("🔍 View Context Sources", expanded=False):
                        tags = "".join(
                            f'<span class="tag">📖 {s["topic"].replace("_", " ").title()}</span>'
                            for s in msg["sources"])
                        st.markdown(f'<div style="margin-bottom:8px">{tags}</div>',
                                    unsafe_allow_html=True)
                        st.caption("Information retrieved from Knowledge Base to formulate this answer.")

                clean = re.sub(r'\$\$(.+?)\$\$', r'\1', msg["content"], flags=re.DOTALL)
                clean = re.sub(r'\$(.+?)\$', r'\1', clean)
                clean = re.sub(r'━+', '─────────────────', clean)
                with st.expander("📋 Copy plain text", expanded=False):
                    st.code(clean, language=None)
                
                chat_export = f"### Assistant Response\n\n{msg['content']}"
                st.download_button(
                    label="📥 Download Answer as Markdown",
                    data=chat_export,
                    file_name=f"math_answer_{i}.md",
                    mime="text/markdown",
                    key=f"dl_btn_{i}"
                )
    
    st.markdown("<br>", unsafe_allow_html=True)

    user_input = st.chat_input("e.g., Solve x² - 5x + 6 = 0 or Find the derivative of sin(x)·x²")
    pending_query = st.session_state.get("pending")

    if pending_query:
        user_input = pending_query
        st.session_state.pending = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.query_count += 1
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧮 Computing solution..."):
                result = st.session_state.engine.query(user_input)
                st.markdown(result["answer"])
                
                if result.get("sources"):
                    with st.expander("🔍 View Context Sources", expanded=False):
                        tags = "".join(
                            f'<span class="tag">📖 {s["topic"].replace("_", " ").title()}</span>'
                            for s in result["sources"])
                        st.markdown(f'<div style="margin-bottom:8px">{tags}</div>',
                                    unsafe_allow_html=True)
                        st.caption("Information retrieved from Knowledge Base to formulate this answer.")
                
                clean = re.sub(r'\$\$(.+?)\$\$', r'\1', result["answer"], flags=re.DOTALL)
                clean = re.sub(r'\$(.+?)\$', r'\1', clean)
                clean = re.sub(r'━+', '─────────────────', clean)
                with st.expander("📋 Copy plain text", expanded=False):
                    st.code(clean, language=None)
                
                chat_export = f"### Assistant Response\n\n{result['answer']}"
                st.download_button(
                    label="📥 Download Answer as Markdown",
                    data=chat_export,
                    file_name="math_answer_latest.md",
                    mime="text/markdown",
                    key=f"dl_btn_latest_{st.session_state.query_count}"
                )

        st.session_state.messages.append({
            "role":          "assistant",
            "content":       result["answer"],
            "sources":       result.get("sources", []),
            "symbolic_hint": result.get("symbolic_hint"),
            "graph_expr":    result.get("graph_expr"),
        })

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
