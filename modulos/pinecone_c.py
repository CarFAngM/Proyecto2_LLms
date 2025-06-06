import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def init_pinecone(index_name: str):
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key or not index_name:
        raise EnvironmentError("Faltan las variables de entorno PINECONE_API_KEY o INDEX_NAME.")

    pc = Pinecone(api_key=api_key)
    index_list = pc.list_indexes().names()

    if index_name not in index_list:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Índice '{index_name}' creado correctamente.")
    else:
        print(f"Índice '{index_name}' ya existe.")

    index = pc.Index(name=index_name)
    return index






