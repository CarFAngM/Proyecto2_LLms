import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone as PineconeClient
from modulos.pinecone_c import init_pinecone

def load_and_store_documents(directory_path: str, index_name: str):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        raise EnvironmentError("La clave OPENAI_API_KEY no está definida en el archivo .env")

    index = init_pinecone()  

    all_docs = []

    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"El directorio '{directory_path}' no existe.")
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        try:
            if filename.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
            elif filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                print(f"Archivo ignorado (extensión no compatible): {filename}")
                continue

            docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = splitter.split_documents(docs)

            for doc in split_docs:
                doc.metadata.setdefault('source', filename)

            all_docs.extend(split_docs)

        except Exception as e:
            print(f"Error al procesar '{filename}': {e}")
            continue

    if not all_docs:
        raise ValueError("No se encontraron documentos válidos para cargar.")
    

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    batch_size = 100
    vectors = []

    for i in range(0, len(all_docs), batch_size):
        batch_docs = all_docs[i:i+batch_size]
        texts = [doc.page_content for doc in batch_docs]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        for j, embedding in enumerate(response.data):
            vector = {
                "id": str(uuid.uuid4()),
                "values": embedding.embedding,
                "metadata": batch_docs[j].metadata
            }
            vectors.append(vector)

    index.upsert(vectors=vectors)
    print(f"Se cargaron {len(vectors)} fragmentos de documentos a Pinecone.")

if __name__ == "__main__":
    load_and_store_documents(directory_path="./docs", index_name="proyecto-index2")