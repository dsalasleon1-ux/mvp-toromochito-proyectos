import os
import tempfile
import streamlit as st
from google import genai
from google.genai import types

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Asistente Toromochito 2.0",
    page_icon="🤖",
    layout="wide",
)

# ============================================================
# API KEY (desde Streamlit Secrets)
# ============================================================
def _get_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", "")

API_KEY = _get_api_key()
client = genai.Client(api_key=API_KEY) if API_KEY else None

MODEL_ID = "gemini-flash-latest"

# ============================================================
# SYSTEM PROMPT CORPORATIVO
# ============================================================
SYSTEM_PROMPT = """Eres Toromochito, el asistente virtual de Minera Chinalco Perú S.A. para el área de Compras y Contratos Proyectos.
Tu función es responder exclusivamente consultas sobre los procedimientos basándote SOLAMENTE en el contenido de los manuales y documentos normativos proporcionados.
Si la respuesta no está en los documentos, indícale al usuario que contacte al comprador responsable.
Nunca inventes información ni compartas datos comerciales confidenciales. 
Bajo el principio Human-in-the-Loop, asistes técnicamente al equipo de proyectos y proveedores bajo los estándares de cumplimiento minero."""

# ============================================================
# HELPER: Subir PDFs locales a Gemini con Mime Type explícito
# ============================================================
def _cargar_docs_locales():
    docs_paths = []
    if os.path.exists("docs"):
        for archivo in os.listdir("docs"):
            if archivo.lower().endswith(".pdf"):
                docs_paths.append(os.path.join("docs", archivo))
    return docs_paths

def _subir_archivos_a_gemini(paths):
    archivos_subidos = []
    for path in paths:
        with open(path, "rb") as f:
            # Especificamos explícitamente que es un archivo PDF para evitar el error de tipo
            file_obj = client.files.upload(
                file=f,
                config=types.UploadFileConfig(mime_type="application/pdf")
            )
            archivos_subidos.append(file_obj)
    return archivos_subidos

# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.markdown("# 🤖 Asistente Toromochito 2.0 (Prueba en Vivo)")
st.markdown(
    "**Minera Chinalco Perú S.A.** · Área de Compras y Contratos Proyectos · "
    "Motor Cognitivo Multimodal Nativo"
)
st.divider()

if client is None:
    st.error("⚠️ Falta configurar la clave `GOOGLE_API_KEY` en Streamlit Secrets.")
    st.stop()

archivos_locales = _cargar_docs_locales()
if not archivos_locales:
    st.error("⚠️ No se encontraron archivos PDF en la carpeta 'docs' de tu repositorio.")
    st.stop()

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
        with st.spinner("Consultando la base de conocimiento y manuales de proyectos..."):
            try:
                archivos_subidos = _subir_archivos_a_gemini(archivos_locales)
                
                contenido_envio = [
                    f"{SYSTEM_PROMPT}\n\nConsulta del usuario: {prompt_usuario}"
                ] + archivos_subidos

                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=contenido_envio,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=4000,
                    ),
                )
                
                texto_respuesta = response.candidates[0].content.parts[0].text
                contenedor_respuesta.markdown(texto_respuesta)
                respuesta_final = texto_respuesta
                
            except Exception as e:
                respuesta_final = f"Ocurrió un error al procesar la consulta con Gemini: {e}"
                contenedor_respuesta.markdown(respuesta_final)
            
    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
