import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuración de la página
st.set_page_config(page_title="Toromochito 2.0 MVP", page_icon="🤖", layout="wide")
st.title("🤖 Asistente Toromochito 2.0 (Prueba en Vivo)")

# 2. Configurar API Key desde los secretos de Streamlit
if "GOOGLE_API_KEY" not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except:
        st.warning("Falta configurar GOOGLE_API_KEY en los Secrets de Streamlit.")
        st.stop()

# 3. Función RAG (Caché para no vectorizar los PDFs con cada mensaje)
@st.cache_resource(show_spinner=False)
def inicializar_motor_rag():
    # Cargar los documentos PDF desde la carpeta 'docs'
    loader = PyPDFDirectoryLoader("docs")
    docs = loader.load()
    
    # Dividir textos en fragmentos legibles por la IA
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # Vectorizar usando los embeddings gratuitos de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Configurar el modelo principal (Gemini 1.5 Flash)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # System Prompt de Gobierno (Extraído de tu Trabajo Final)
    system_prompt = (
        "Eres Toromochito, el asistente virtual de Minera Chinalco Perú S.A. para el área de Compras y Contratos Proyectos. "
        "Tu función es responder exclusivamente consultas sobre los procedimientos basándote SOLAMENTE en el contexto proporcionado. "
        "Si la respuesta no está en tu base de conocimiento, indica al usuario que contacte al comprador responsable. "
        "Nunca inventes información ni compartas datos comerciales confidenciales.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Integrar el recuperador de texto con el modelo
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)
    
    return rag_chain

# Validar existencia de manuales e inicializar
if os.path.exists("docs") and os.listdir("docs"):
    with st.spinner("Procesando manuales de Compras y Contratos..."):
        rag_chain = inicializar_motor_rag()
else:
    st.error("Por favor, crea una carpeta llamada 'docs' y sube al menos un documento PDF.")
    st.stop()

# 4. Interfaz de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Dibujar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar consulta del usuario
if prompt_usuario := st.chat_input("Escribe tu consulta (Ej. ¿Cuáles son los pasos para habilitar personal en TESEO?)"):
    
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)

    # Procesar respuesta con IA
    with st.chat_message("assistant"):
        contenedor_respuesta = st.empty()
        with st.spinner("Consultando la base de conocimiento..."):
            # Invocar al motor RAG
            response = rag_chain.invoke({"input": prompt_usuario})
            respuesta_final = response["answer"]
            
            # Mostrar la respuesta
            contenedor_respuesta.markdown(respuesta_final)
            
    # Guardar en el historial
    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})