import os
import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "proyecto-index2"

def initialize_pinecone():
    """Inicializa y verifica la conexión con Pinecone"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if INDEX_NAME not in pc.list_indexes().names():
            st.error(f"El índice '{INDEX_NAME}' no existe en Pinecone")
            return None
        return pc.Index(INDEX_NAME)
    except Exception as e:
        st.error(f"Error conectando a Pinecone: {str(e)}")
        return None

def retrieve_context(query, index):
    """Recupera contexto relevante de Pinecone"""
    try:

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        vector = embeddings.embed_query(query)
        

        results = index.query(
            vector=vector,
            top_k=3,
            include_metadata=True
        )
        
        context_chunks = []
        for match in results.matches:
            metadata = match.metadata or {}
            source = metadata.get('source', '').replace('\\', '/')
            page = metadata.get('page', 'N/A')
            
            context_chunks.append(
                f"Documento: {source}\n"
                f"Página: {page}\n"
                f"ID: {match.id}\n"
                "------------------"
            )
        
        return "\n".join(context_chunks) if context_chunks else "No se encontró contexto relevante."
    
    except Exception as e:
        st.error(f"Error buscando contexto: {str(e)}")
        return ""

def generate_response(query, context):
    """Genera una respuesta usando el contexto obtenido"""
    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )

        prompt = (
            "Responde la pregunta basándote únicamente en el contexto proporcionado.\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pregunta: {query}\n\n"
            "Respuesta:"
        )
        
        response = llm.invoke(prompt)
        return response.content
    
    except Exception as e:
        st.error(f"Error generando respuesta: {str(e)}")
        return ""

def main():
    st.title("Sistema de Consulta Documental")
    

    if 'pinecone_index' not in st.session_state:
        st.session_state.pinecone_index = initialize_pinecone()
    
    if st.session_state.pinecone_index:
        user_query = st.text_input("Ingresa tu pregunta sobre los documentos:")
        
        if user_query:
            with st.spinner("Buscando en los documentos..."):

                context = retrieve_context(user_query, st.session_state.pinecone_index)
                
                if context and "No se encontró" not in context:

                    st.subheader("📄 Documentos encontrados")
                    st.text_area("Contexto relevante:", value=context, height=200)
                    

                    with st.spinner("Generando respuesta basada en los documentos..."):
                        response = generate_response(user_query, context)
                        st.subheader("🧠 Respuesta")
                        st.write(response)
                else:
                    st.warning("No se encontraron documentos relevantes para tu consulta.")
    else:
        st.error("No se pudo conectar con la base de datos de documentos.")

if __name__ == "__main__":
    main()