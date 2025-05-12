import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv


load_dotenv()


def init_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index_list = pc.list_indexes().names()
    index_name = os.getenv("INDEX_NAME")

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

    index = pc.Index(index_name)  
    return index













