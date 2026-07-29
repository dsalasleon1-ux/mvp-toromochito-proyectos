import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

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

# 3. Función RAG (Método Directo y a Prueba de Fallos)
@st.cache_resource(show_spinner=False)
def inicializar_motor_rag():
    # Cargar y leer los PDFs de los manuales
    loader = PyPDFDirectoryLoader("docs")
    docs = loader.load()
    
    # Dividir el texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # Vectorizar usando el modelo gratuito de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Configurar el motor principal de inteligencia artificial
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # Retornar el buscador de documentos y el motor LLM
    return vectorstore.as_retriever(search_kwargs={"k": 4}), llm

# Inicializar
if os.path.exists("docs") and os.listdir("docs"):
    with st.spinner("Procesando manuales de Compras y Contratos..."):
        retriever, llm = inicializar_motor_rag()
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
            try:
                # PASO A: Buscar información relevante en tus PDFs
                documentos_recuperados = retriever.invoke(prompt_usuario)
                contexto_text = "\n\n".join([doc.page_content for doc in documentos_recuperados])
                
                # PASO B: Inyectar la información al System Prompt corporativo
                prompt_final = f"""Eres Toromochito, el asistente virtual de Minera Chinalco Perú S.A. para el área de Compras y Contratos Proyectos.
                Tu función es responder exclusivamente consultas sobre los procedimientos basándote SOLAMENTE en el contexto proporcionado.
                Si la respuesta no está en tu base de conocimiento, indica al usuario que contacte al comprador responsable.
                Nunca inventes información ni compartas datos comerciales confidenciales.

                DOCUMENTOS DE CONTEXTO ENCONTRADOS:
                {contexto_text}

                CONSULTA DEL PROVEEDOR: {prompt_usuario}
                
                RESPUESTA:"""
                
                # PASO C: Generar la respuesta final
                response = llm.invoke(prompt_final)
                respuesta_final = response.content
                contenedor_respuesta.markdown(respuesta_final)
                
            except Exception as e:
                respuesta_final = f"Ocurrió un error en la nube: {e}"
                contenedor_respuesta.markdown(respuesta_final)
            
    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
