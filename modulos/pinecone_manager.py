"""
Gestión de Pinecone en modo funcional con chunking por página y deduplicación.
"""

import os
import time
from typing import Dict, List, Optional
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

_pinecone_client: Optional[Pinecone] = None
_pinecone_index = None


def _get_client() -> Pinecone:
    global _pinecone_client
    if _pinecone_client:
        return _pinecone_client
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Se requiere PINECONE_API_KEY")
    _pinecone_client = Pinecone(api_key=api_key)
    return _pinecone_client


def _get_index_name() -> str:
    return os.getenv("INDEX_NAME", "cv-database")


def create_index_if_needed(dimension: int) -> None:
    global _pinecone_index
    client = _get_client()
    index_name = _get_index_name()

    if index_name not in client.list_indexes().names():
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(5)  # esperar disponibilidad

    _pinecone_index = client.Index(name=index_name)


def get_index(dimension: int):
    global _pinecone_index
    if _pinecone_index is None:
        create_index_if_needed(dimension)
    return _pinecone_index


def check_duplicate(document_hash: str, dimension: int) -> bool:
    try:
        index = get_index(dimension)
        results = index.query(
            vector=[0.0] * dimension,
            filter={"document_hash": {"$eq": document_hash}},
            top_k=1,
            include_metadata=True,
        )
        return len(results.get("matches", [])) > 0
    except Exception as exc:  # pragma: no cover
        print(f"Error verificando duplicados: {exc}")
        return False


def upsert_cv_pages(
    cv_id: str,
    embeddings: List[List[float]],
    metadata: Dict,
    page_texts: List[str],
    dimension: int,
) -> bool:
    try:
        index = get_index(dimension)

        document_hash = metadata.get("document_hash")
        if document_hash and check_duplicate(document_hash, dimension):
            print(f"⚠ CV duplicado encontrado (hash: {document_hash}). No se insertará.")
            return False

        vectors = []
        total_pages = metadata.get("total_pages", len(page_texts))

        for page_number, (embedding, page_text) in enumerate(zip(embeddings, page_texts), start=1):
            page_metadata = metadata.copy()
            page_metadata.update({
                "cv_id": cv_id,
                "page_number": page_number,
                "total_pages": total_pages,
                "page_word_count": len(page_text.split()),
                "page_char_count": len(page_text),
                "cv_text": metadata.get("cv_text", ""),
                "page_text": page_text[:1200],  # limitar tamaño de metadata por vector
            })
            vectors.append({
                "id": f"{cv_id}_p{page_number}",
                "values": embedding,
                "metadata": page_metadata,
            })

        index.upsert(vectors=vectors)
        print(f"✓ CV insertado en {len(vectors)} páginas: {cv_id}")
        return True
    except Exception as exc:  # pragma: no cover
        print(f"Error insertando CV: {exc}")
        return False


def search_similar_cvs(query_embedding: List[float], top_k: int, dimension: int) -> List[Dict]:
    try:
        index = get_index(dimension)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        grouped: Dict[str, Dict] = {}
        for match in results.get("matches", []):
            meta = match.get("metadata", {}) or {}
            cv_id = meta.get("cv_id") or meta.get("document_hash") or match.get("id")
            if not cv_id:
                continue

            group = grouped.setdefault(cv_id, {
                "id": cv_id,
                "scores": [],
                "metadata": meta,
                "page_matches": [],
            })

            group["scores"].append(match.get("score", 0))
            group["page_matches"].append({
                "page_number": meta.get("page_number"),
                "score": match.get("score", 0),
            })

        aggregated: List[Dict] = []
        for group in grouped.values():
            scores = group["scores"]
            avg_score = sum(scores) / len(scores) if scores else 0
            aggregated.append({
                "id": group["id"],
                "score": avg_score,
                "metadata": group["metadata"],
                "page_matches": group["page_matches"],
            })

        aggregated.sort(key=lambda item: item["score"], reverse=True)
        return aggregated
    except Exception as exc:  # pragma: no cover
        raise Exception(f"Error buscando CVs similares: {exc}")


def clear_index(dimension: int) -> bool:
    try:
        index = get_index(dimension)
        index.delete(delete_all=True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"Error limpiando índice: {exc}")
        return False
