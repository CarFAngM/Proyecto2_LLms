"""
Módulos del Sistema RAG para Análisis de Currículums
"""

from .pdf_processor import process_cv, clean_text, extract_text_from_pdf
from .embedding_generator import generate_embedding, generate_embeddings_batch, get_embedding_dimension
from .pinecone_manager import (
    create_index_if_needed,
    check_duplicate,
    upsert_cv_pages,
    search_similar_cvs,
    clear_index,
)
from .profile_matcher import analyze_candidate, rank_candidates, generate_comparison_report

__all__ = [
    'process_cv',
    'clean_text',
    'extract_text_from_pdf',
    'generate_embedding',
    'generate_embeddings_batch',
    'get_embedding_dimension',
    'create_index_if_needed',
    'check_duplicate',
    'upsert_cv_pages',
    'search_similar_cvs',
    'clear_index',
    'analyze_candidate',
    'rank_candidates',
    'generate_comparison_report',
]

__version__ = '1.0.0'
