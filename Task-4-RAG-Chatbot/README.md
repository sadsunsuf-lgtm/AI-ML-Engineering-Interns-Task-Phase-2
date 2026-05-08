🤖 Task 4: Agentic PDF RAG Chatbot
An advanced Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and engage in a context-aware conversation. Built using the LangChain ecosystem and powered by the Llama-3.1-8b model via Groq for lightning-fast inference.

🚀 Live Demo
https://huggingface.co/spaces/SanaNasir/agentic-rag-pdf

✨ Key Features
Intelligent PDF Parsing: Automatically extracts and chunks text from uploaded PDFs using PyPDFLoader.

Agentic Memory: Maintains conversation history using ConversationBufferMemory, allowing for natural follow-up questions.

High-Speed RAG: Utilizes FAISS (Facebook AI Similarity Search) and HuggingFaceEmbeddings for sub-second retrieval.

Professional UI: A sleek, custom-styled Streamlit interface with high-visibility components designed for ease of use.

Global Content Intelligence: Capable of auditing and grading document quality in real-time.

🛠️ Tech Stack
Language: Python 3.11

Framework: Streamlit

Orchestration: LangChain (using langchain-classic and langchain-text-splitters)

LLM Engine: Groq (Llama-3.1-8b-instant)

Vector Database: FAISS (CPU)

Embeddings: HuggingFace all-MiniLM-L6-v2

Deployment: Docker on Hugging Face Spaces

📂 Project Structure
Plaintext
Task 4/
├── app.py                # Main Streamlit Application logic
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration for deployment
└── .streamlit/
    └── secrets.toml      # Local API keys (Excluded from Git)
⚙️ Local Installation & Setup
Clone the repository:

Bash
git clone https://github.com/sadsunsuf-lgtm/AI-ML-Engineering-Interns-Task-Phase-2.git
cd AI-ML-Engineering-Interns-Task-Phase-2/"Task 4"
Set up a virtual environment:

Bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Configure Secrets:
Create a .streamlit/secrets.toml file and add your Groq API key:

Ini, TOML
GROQ_API_KEY = "your_key_here"
Run the App:

Bash
streamlit run app.py
🛡️ Deployment Notes
This project is containerized using Docker for consistent behavior across environments. For production deployment on Hugging Face, ensure the app_port is set to 7860 in the Space metadata and the GROQ_API_KEY is added as a Secret in the Space settings.
