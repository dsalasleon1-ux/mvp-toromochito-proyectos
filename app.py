import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
# LÍNEA CORREGIDA:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Configuración de la página
st.set_page_config(page_title="Toromochito 2.0 MVP", page_icon="🤖", layout="wide")
st.title("🤖 Asistente Toromochito 2.0 (Prueba en Vivo)")

# 2. Configurar API Key
if "GOOGLE_API_KEY" not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except:
        st.warning("Falta configurar GOOGLE_API_KEY en los Secrets de Streamlit.")
        st.stop()

# 3. Función RAG (Método Universal)
@st.cache_resource(show_spinner=False)
def inicializar_motor_rag():
    loader = PyPDFDirectoryLoader("docs")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # System Prompt de Gobierno
    prompt_template = """Eres Toromochito, el asistente virtual de Minera Chinalco Perú S.A. para el área de Compras y Contratos Proyectos.
    Tu función es responder exclusivamente consultas sobre los procedimientos basándote SOLAMENTE en el contexto proporcionado.
    Si la respuesta no está en tu base de conocimiento, indica al usuario que contacte al comprador responsable.
    Nunca inventes información ni compartas datos comerciales confidenciales.

    Contexto: {context}

    Consulta: {question}
    Respuesta:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Uso de RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Inicializar
if os.path.exists("docs") and os.listdir("docs"):
    with st.spinner("Procesando manuales de Compras y Contratos..."):
        qa_chain = inicializar_motor_rag()
else:
    st.error("Por favor, verifica que la carpeta 'docs' tenga los manuales.")
    st.stop()

# 4. Interfaz de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_usuario := st.chat_input("Escribe tu consulta (Ej. Pasos para habilitar personal en TESEO)"):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)

    with st.chat_message("assistant"):
        contenedor_respuesta = st.empty()
        with st.spinner("Consultando la base de conocimiento..."):
            response = qa_chain.invoke({"query": prompt_usuario})
            respuesta_final = response["result"]
            contenedor_respuesta.markdown(respuesta_final)
            
    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
