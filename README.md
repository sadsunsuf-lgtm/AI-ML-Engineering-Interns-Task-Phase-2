# 🚀 AI & ML Engineering Internship: Phase 2 Portfolio

Welcome to my Phase 2 project repository. This collection showcases end-to-end development in Artificial Intelligence, ranging from transformer-based classification and automated ML pipelines to advanced Agentic RAG systems.

---

## 👤 About the Developer
I am an **Independent AI Automation Developer** based in Karachi, Pakistan. I specialize in building production-ready AI applications using Python, LangChain, and various LLM orchestrations.

---

## 📚 Table of Contents
1. [Task 1: News Topic Classifier (BERT)](#task-1-news-topic-classifier-using-bert)
2. [Task 2: Customer Churn Pipeline (Scikit-Learn)](#task-2-end-to-end-ml-pipeline)
3. [Task 3: Multimodal Housing Price Prediction (CNN + MLP)](#task-3-multimodal-ai-housing-price-prediction)
4. [Task 4: Agentic PDF RAG Chatbot (LangChain + Groq)](#task-4-agentic-pdf-rag-chatbot)
5. [Task 5: Auto-Tagging Support Tickets (LLM Prompt Engineering)](#task-5-auto-tagging-support-tickets-using-llm)

---

## 📰 Task 1: News Topic Classifier Using BERT
**[🚀 Live Demo](https://huggingface.co/spaces/SanaNasir/News-Topic-Classifier)**

* **Objective:** Categorize news headlines into World, Sports, Business, and Sci/Tech using Transfer Learning.
* **Tech:** BERT (`bert-base-uncased`), Hugging Face Transformers, Gradio.
* **Key Insight:** Fine-tuning BERT outperformed traditional ML by understanding linguistic context rather than just word frequency.

---

## 📈 Task 2: End-to-End ML Pipeline
* **Objective:** Build a production-ready pipeline for predicting customer churn with seamless transition from training to inference.
* **Tech:** Scikit-learn, Logistic Regression, GridSearchCV, Joblib.
* **Key Insight:** Contract type and monthly charges were the strongest churn predictors.

---

## 🏠 Task 3: Multimodal AI: Housing Price Prediction
* **Objective:** Predict real estate values by fusing structured numerical data with visual architectural features (images).
* **Tech:** Keras, CNN (for images), Feed-Forward Neural Network (for tabular), Feature Fusion.
* **Key Insight:** Combining visual "curb appeal" data with numerical stats significantly reduced prediction error.

---

## 🤖 Task 4: Agentic PDF RAG Chatbot
**[🚀 Live Demo](https://huggingface.co/spaces/SanaNasir/agentic-rag-pdf)**

* **Objective:** A context-aware system allowing users to chat with uploaded PDFs using Retrieval-Augmented Generation.
* **Tech:** LangChain, Llama-3.1-8b (Groq), FAISS Vector DB, Streamlit, Docker.
* **Key Features:** `ConversationBufferMemory` for multi-turn dialogue and sub-second retrieval.

---

## 🎫 Task 5: Auto Tagging Support Tickets using LLM
* **Objective:** Automate ticket routing by predicting the Top 3 most probable categories for customer support queries.
* **Tech:** Llama-3.1-8b, Prompt Engineering (Zero-Shot vs. Few-Shot), Groq API.
* **Key Observation:** Few-shot learning drastically improved formatting and accuracy compared to zero-shot attempts.

---

## 🛠️ Global Tech Stack
| Category | Tools |
| :--- | :--- |
| **Languages** | Python 3.11 |
| **AI/ML** | BERT, Scikit-learn, Keras, LangChain, FAISS |
| **LLMs** | Llama 3.1 (via Groq), OpenAI API |
| **Deployment** | Hugging Face Spaces, Docker, Streamlit, Gradio |

---

## 📂 Repository Structure
```text
.
├── Task 1/              # News Topic Classifier (BERT)
├── Task 2/              # Customer Churn Pipeline
├── Task 3/              # Multimodal Housing Prediction
├── Task 4/              # Agentic PDF RAG Chatbot
├── Task 5/              # LLM Ticket Tagging
└── README.md            # Main Portfolio Documentation
