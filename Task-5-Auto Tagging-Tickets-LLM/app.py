import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Agentic PDF AI", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stSidebar"] { background-color: #1e1e2f; color: white; }
    
    [data-testid="stFileUploadDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #2575fc !important;
        border-radius: 10px;
        padding: 20px;
    }
    
    [data-testid="stFileUploadDropzone"] div div span, 
    [data-testid="stFileUploadDropzone"] div div small,
    [data-testid="stFileUploadDropzone"] label {
        color: #000000 !important;
        font-weight: bold !important;
    }
    [data-testid="stBaseButton-secondary"] {
        color: #ffffff !important; 
        background-color: #2575fc !important;
        border-radius: 10px !important;
        border: none !important;
    }
    .stButton>button {
        background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
        color: white !important;
        border-radius: 20px;
        border: none;
        font-weight: bold;
    }
    
    .stMarkdown h1 { color: #1e1e2f; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SETTINGS & API KEY (DOCKER COMPATIBLE) ---
# We check the OS environment first (standard for Docker/Hugging Face)
key = os.environ.get("GROQ_API_KEY")

# If not in OS, we check st.secrets ONLY if the file actually exists
if not key:
    try:
        if "GROQ_API_KEY" in st.secrets:
            key = st.secrets["GROQ_API_KEY"]
    except Exception:
        # If secrets.toml is missing, just ignore the error and move on
        pass

if key:
    os.environ["GROQ_API_KEY"] = key
else:
    st.error("🔑 API Key not found! Go to Space Settings > Secrets and add GROQ_API_KEY.")
    st.stop()

# --- 3. DYNAMIC PDF PROCESSING ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_new_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(uploaded_file.getbuffer())
        file_path = tf.name
    
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        embeddings = get_embeddings()
        vector_db = FAISS.from_documents(splits, embeddings)
        return vector_db
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("AI Settings")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload a PDF to Chat", type="pdf")
    
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        st.rerun()

# --- 5. CORE LOGIC ---
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if uploaded_file:
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        with st.spinner("Analyzing document..."):
            st.session_state.vector_db = process_new_pdf(uploaded_file)
            st.session_state.last_uploaded = uploaded_file.name
            st.success("Analysis Complete!")
elif os.path.exists("faiss_index_store"):
    embeddings = get_embeddings()
    st.session_state.vector_db = FAISS.load_local("faiss_index_store", embeddings, allow_dangerous_deserialization=True)

# --- 6. CHAT UI ---
st.title("🤖 Agentic PDF Assistant")
st.write("Interact with your data using high-speed RAG.")

if st.session_state.vector_db:
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vector_db.as_retriever(),
        memory=st.session_state.memory
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"question": prompt})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.warning("Please upload a PDF in the sidebar to activate the AI!")
