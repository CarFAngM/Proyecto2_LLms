import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from modulos.pinecone_c import init_pinecone 

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

def initialize_pinecone():
    try:
        index = init_pinecone(INDEX_NAME)
        return index
    except Exception as e:
        st.error(f"Error inicializando Pinecone: {e}")
        return None

def load_documents_from_folder(folder="docs"):
    docs = []
    for file in Path(folder).glob("*.txt"):
        loader = TextLoader(str(file), encoding='utf-8')
        docs.extend(loader.load())
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20
    )
    return splitter.split_documents(docs)

def embed_and_upsert_documents(index, docs):
    if not docs:
        st.warning("No se encontraron documentos para procesar.")
        return 0

    client = OpenAI(api_key=OPENAI_API_KEY)
    texts = [doc.page_content for doc in docs]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    vectors = []
    for i, embedding in enumerate(response.data):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding.embedding,
            "metadata": {
                "text": texts[i],
                "source": docs[i].metadata.get('source', 'unknown')
            }
        })

    index.upsert(vectors=vectors)
    return len(vectors)

def retrieve_context(query, index):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        embeddings = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        vector = embeddings.data[0].embedding

        results = index.query(
            vector=vector,
            top_k=3,
            include_metadata=True
        )

        context_chunks = []
        for match in results.matches:
            metadata = match.metadata or {}
            source = metadata.get('source', '').replace('\\', '/')
            text = metadata.get('text', '')
            context_chunks.append(
                f"Fuente: {source}\nTexto: {text}\n------------------"
            )
        return "\n".join(context_chunks) if context_chunks else "No se encontró contexto relevante."
    except Exception as e:
        st.error(f"Error buscando contexto: {str(e)}")
        return ""

def generate_response(query, context):
    try:
        from langchain_openai import ChatOpenAI
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
        index = st.session_state.pinecone_index


        if st.button("Cargar y procesar documentos desde carpeta 'docs'"):
            with st.spinner("Cargando documentos y creando embeddings..."):
                docs = load_documents_from_folder(folder= r"C:\Users\carlo\OneDrive\Escritorio\Proyecto_2 llm\Proyecto2_LLms\docs")
                split_docs = split_documents(docs)
                inserted = embed_and_upsert_documents(index, split_docs)
                st.success(f"Se insertaron {inserted} fragmentos en Pinecone.")

        user_query = st.text_input("Ingresa tu pregunta sobre los documentos:")

        if user_query:
            with st.spinner("Buscando en los documentos..."):
                context = retrieve_context(user_query, index)

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
