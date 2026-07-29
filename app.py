import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Toromochito 2.0 MVP", page_icon="🤖", layout="wide")
st.title("🤖 Asistente Toromochito 2.0 (Prueba en Vivo)")

# Verificación de la llave a nivel de sistema (Nativo)
if "GOOGLE_API_KEY" not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except:
        st.error("Falta configurar GOOGLE_API_KEY en los Secrets de Streamlit.")
        st.stop()

@st.cache_resource(show_spinner=False)
def inicializar_motor_rag():
    try:
        # 1. Leer los manuales
        loader = PyPDFDirectoryLoader("docs")
        docs = loader.load()
        
        if not docs:
            return None, None, "Error: La carpeta 'docs' está vacía o contiene archivos sin texto leíble."
        
        # 2. Cortar los textos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        if not splits:
            return None, None, "Error: No se pudo fragmentar el texto de los manuales."

        # 3. Vectorizar dejando que la librería tome la llave del sistema automáticamente
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # 4. Configurar la IA 
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        return vectorstore.as_retriever(search_kwargs={"k": 4}), llm, "OK"
    except Exception as e:
        return None, None, f"Error de conexión con Google: {str(e)}"

# Inicialización
if os.path.exists("docs") and os.listdir("docs"):
    with st.spinner("Procesando manuales de Compras y Contratos..."):
        retriever, llm, status = inicializar_motor_rag()
        if status != "OK":
            st.error(status)
            st.stop()
else:
    st.error("Por favor, verifica que la carpeta 'docs' tenga los manuales.")
    st.stop()

# Interfaz de Chat
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
                documentos_recuperados = retriever.invoke(prompt_usuario)
                contexto_text = "\n\n".join([doc.page_content for doc in documentos_recuperados])
                
                prompt_final = f"""Eres Toromochito, el asistente virtual de Minera Chinalco Perú S.A. para el área de Compras y Contratos Proyectos.
                Tu función es responder exclusivamente consultas sobre los procedimientos basándote SOLAMENTE en el contexto proporcionado.
                Nunca inventes información ni compartas datos comerciales confidenciales.

                CONTEXTO:
                {contexto_text}

                CONSULTA: {prompt_usuario}
                
                RESPUESTA:"""
                
                response = llm.invoke(prompt_final)
                respuesta_final = response.content
                contenedor_respuesta.markdown(respuesta_final)
                
            except Exception as e:
                respuesta_final = f"Ocurrió un error en la generación: {e}"
                contenedor_respuesta.markdown(respuesta_final)
            
    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
