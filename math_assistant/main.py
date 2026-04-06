import sys
import argparse
import logging
from dotenv import load_dotenv

# Load env before any imports
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("math_assistant")

def _is_streamlit():
    """Detect if we are running inside a Streamlit runtime."""
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False

def main():
    if any("streamlit" in arg for arg in sys.argv):
        from math_assistant.app.ui import run_streamlit_app
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
        from math_assistant.evaluation.eval import run_tests
        sys.exit(0 if run_tests() else 1)
    elif args.eval:
        from math_assistant.evaluation.eval import run_evaluation
        run_evaluation()
    elif args.setup or args.rebuild:
        from math_assistant.knowledge.build_kb import build_pipeline
        store = build_pipeline(force_rebuild=args.rebuild)
        print(f"\n✅ Knowledge base ready — {store.get_document_count()} chunks indexed")
    else:
        print("\nTo launch the UI:\n")
        print("  python3.11 -m streamlit run math_assistant/main.py\n")

if _is_streamlit():
    from math_assistant.app.ui import run_streamlit_app
    run_streamlit_app()
elif __name__ == "__main__":
    main()
