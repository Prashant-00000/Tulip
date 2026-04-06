## Tech Stack
- LLM: Groq API (llama3)
- Vector Store: FAISS
- Math Engine: SymPy
- Embeddings: Sentence Transformers
- Language: Python 3.10

## Setup & Installation

1. Clone the repository
   git clone https://github.com/Prashant-00000/Tulip.git
   cd Tulip

2. Install dependencies
   pip install -r requirements.txt

3. Configure environment
   cp env.example .env
   # Add your API keys to .env

4. Run the assistant
   python math_assistant/main.py

## Environment Variables
See env.example for required API keys and configuration.

## Use Cases
- Students solving math homework
- Learning step-by-step solutions
- Quick formula lookups
- Complex equation solving
