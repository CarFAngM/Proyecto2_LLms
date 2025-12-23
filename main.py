"""
Sistema RAG para Análisis de Currículums
Interfaz de usuario con Streamlit (modo funcional, chunk por página).
"""

import os
import streamlit as st
import uuid
from dotenv import load_dotenv

# Módulos funcionales
from modulos.pdf_processor import process_cv
from modulos.embedding_generator import (
    generate_embedding,
    generate_embeddings_batch,
    get_embedding_dimension,
)
from modulos.pinecone_manager import (
    create_index_if_needed,
    check_duplicate,
    upsert_cv_pages,
    search_similar_cvs,
    clear_index,
)
from modulos.profile_matcher import rank_candidates, generate_comparison_report

load_dotenv()

st.set_page_config(
    page_title="Sistema RAG - Análisis de CVs",
    page_icon="",
    layout="wide"
)

if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "embedding_dimension" not in st.session_state:
    st.session_state.embedding_dimension = None
if "last_ranked_candidates" not in st.session_state:
    st.session_state.last_ranked_candidates = []
if "last_ideal_profile" not in st.session_state:
    st.session_state.last_ideal_profile = ""


def initialize_system():
    """Inicializa cliente de embeddings y crea índice Pinecone."""
    try:
        with st.spinner("Inicializando sistema..."):
            dimension = get_embedding_dimension()
            create_index_if_needed(dimension)
            st.session_state.embedding_dimension = dimension
            st.session_state.initialized = True
            st.success("Sistema inicializado correctamente")
    except Exception as exc:
        st.error(f"Error inicializando sistema: {exc}")
        st.session_state.initialized = False


def process_and_upload_cv(uploaded_file):
    """Procesa un CV, genera embeddings por página y lo sube a Pinecone."""
    try:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        cv_data = process_cv(temp_path)
        if cv_data is None:
            return {"success": False, "message": "Error procesando PDF"}

        document_hash = cv_data["metadata"].get("document_hash")
        dimension = st.session_state.embedding_dimension

        if check_duplicate(document_hash, dimension):
            return {"success": False, "message": "CV duplicado (ya existe en la base de datos)"}

        # Guardar texto completo truncado en metadata para análisis LLM
        cv_text = cv_data.get("text", "")
        metadata_with_text = cv_data["metadata"].copy()
        metadata_with_text["cv_text"] = cv_text[:4000] if len(cv_text) > 4000 else cv_text
        metadata_with_text["full_text_length"] = len(cv_text)

        # Embeddings por página
        page_embeddings = generate_embeddings_batch(cv_data.get("page_texts", []))
        if not page_embeddings:
            return {"success": False, "message": "No se pudieron generar embeddings para las páginas"}

        cv_id = str(uuid.uuid4())
        success = upsert_cv_pages(
            cv_id=cv_id,
            embeddings=page_embeddings,
            metadata=metadata_with_text,
            page_texts=cv_data.get("page_texts", []),
            dimension=dimension,
        )

        os.remove(temp_path)

        if success:
            return {
                "success": True,
                "message": f"CV procesado: {uploaded_file.name}",
                "cv_id": cv_id,
                "metadata": metadata_with_text,
            }
        return {"success": False, "message": "Error insertando en base de datos"}
    except Exception as exc:
        return {"success": False, "message": f"Error: {exc}"}


def main():
    st.title("Sistema RAG - Análisis de Currículums")
    st.markdown("---")

    with st.sidebar:
        st.header("Configuración")
        if not st.session_state.initialized:
            if st.button("Inicializar Sistema", type="primary"):
                initialize_system()
        else:
            st.success("Sistema activo")
            if st.button("Limpiar Base de Datos", type="secondary"):
                if clear_index(st.session_state.embedding_dimension):
                    st.success("Base de datos limpiada")

    if not st.session_state.initialized:
        st.info("Por favor, inicializa el sistema usando el botón en la barra lateral")
        return

    tab1, tab2, tab3 = st.tabs(["Cargar CVs", "Buscar Candidatos", "Rankings"])
    
    with tab1:
        st.header("Cargar Currículums (PDF)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Selecciona archivos PDF de CVs",
                type=['pdf'],
                accept_multiple_files=True,
                help="Puedes seleccionar múltiples archivos PDF"
            )
        
        with col2:
            st.info("""
            Información:
            - Solo archivos PDF
            - Se detectan automáticamente duplicados
            - Se extrae metadata automáticamente
            """)
        
        if uploaded_files:
            if st.button("Procesar y Cargar CVs", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                total_files = len(uploaded_files)
                successful = 0
                duplicates = 0
                errors = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Procesando {i+1}/{total_files}: {uploaded_file.name}")
                    
                    result = process_and_upload_cv(uploaded_file)
                    
                    with results_container:
                        if result["success"]:
                            metadata = result.get('metadata', {})
                            word_count = metadata.get('word_count', 0)
                            st.success(f"{result['message']} ({word_count} palabras)")
                            successful += 1
                        else:
                            if "duplicado" in result["message"].lower():
                                st.warning(f"{uploaded_file.name}: {result['message']}")
                                duplicates += 1
                            else:
                                st.error(f"{uploaded_file.name}: {result['message']}")
                                errors += 1
                    
                    progress_bar.progress((i + 1) / total_files)
                
                status_text.empty()
                progress_bar.empty()
                
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total procesados", total_files)
                with col2:
                    st.metric("Exitosos", successful)
                with col3:
                    st.metric("Duplicados", duplicates)
                with col4:
                    st.metric("Errores", errors)
    
    with tab2:
        st.header("Buscar Candidatos por Perfil Ideal")

        ideal_profile = st.text_area(
            "Describe el perfil ideal del candidato",
            height=200,
            placeholder="""Ejemplo:
Buscamos un desarrollador Full Stack con experiencia en:
- Python y JavaScript
- Frameworks: React, Node.js, Django
- Bases de datos: PostgreSQL, MongoDB
- Experiencia mínima: 3 años
- Conocimientos en AWS y DevOps
- Inglés avanzado
""",
            help="Describe las habilidades, experiencia y requisitos del candidato ideal",
        )

        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Número de candidatos a analizar", 5, 150, 10)
        with col2:
            min_score = st.slider("Score mínimo (0-100)", 0, 100, 70)

        st.info(
            """
            Escala de Evaluación (ESTRICTA):
            - 90-100: Excepcional (cumple todo + excede)
            - 75-89: Excelente (cumple todos los requisitos)
            - 60-74: Bueno (cumple mayoría, faltan 1-2 habilidades)
            - 40-59: Regular (faltan habilidades críticas)
            - 20-39: Débil (perfil muy diferente)
            - 0-19: Inadecuado (rol completamente diferente)
            """
        )

        if st.button("Buscar Mejores Candidatos", type="primary", disabled=not ideal_profile.strip()):
            with st.spinner("Analizando candidatos..."):
                try:
                    profile_embedding = generate_embedding(ideal_profile)
                    similar_cvs = search_similar_cvs(
                        profile_embedding,
                        top_k=top_k,
                        dimension=st.session_state.embedding_dimension
                    )

                    if not similar_cvs:
                        st.warning("No se encontraron candidatos en la base de datos")
                    else:
                        st.info(f"Encontrados {len(similar_cvs)} candidatos similares. Iniciando análisis detallado...")

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(current, total):
                            progress = current / total
                            progress_bar.progress(progress)
                            status_text.text(f"Analizando candidato {current} de {total}...")

                        ranked_candidates = rank_candidates(
                            ideal_profile,
                            similar_cvs,
                            top_k=top_k,
                            progress_callback=update_progress,
                        )

                        progress_bar.empty()
                        status_text.empty()

                        st.session_state.last_ranked_candidates = ranked_candidates
                        st.session_state.last_ideal_profile = ideal_profile

                        filtered_candidates = [
                            c for c in ranked_candidates
                            if c["analysis"].get("match_score", 0) >= min_score
                        ]

                        if not filtered_candidates:
                            st.warning(f"No se encontraron candidatos con score >= {min_score}")
                        else:
                            st.success(f"Encontrados {len(filtered_candidates)} candidatos que cumplen los criterios")

                            for i, candidate in enumerate(filtered_candidates, 1):
                                analysis = candidate["analysis"]
                                metadata = candidate["metadata"]
                                match_score = analysis.get("match_score", 0)
                                vector_score = candidate.get("vector_score", 0)

                                if match_score >= 90:
                                    color = "green"
                                    icon = ""
                                elif match_score >= 75:
                                    color = "blue"
                                    icon = ""
                                elif match_score >= 60:
                                    color = "orange"
                                    icon = ""
                                elif match_score >= 40:
                                    color = "orange"
                                    icon = ""
                                else:
                                    color = "red"
                                    icon = ""

                                with st.expander(
                                    f"{icon} #{i} - {metadata.get('filename', 'Sin nombre')} - Score: {match_score}/100",
                                    expanded=(i == 1),
                                ):
                                    col1, col2, col3 = st.columns(3)
                                    recommendation = analysis.get("recommendation", "Consider")
                                    with col1:
                                        st.metric("Match Score", f"{match_score}/100")
                                    with col2:
                                        st.metric("Vector Score (promedio)", f"{vector_score:.3f}")
                                    with col3:
                                        st.markdown("Recomendación:")
                                        st.markdown(f":{color}[{recommendation}]")

                                    st.markdown("---")

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("Fortalezas:")
                                        for strength in analysis.get("strengths", []):
                                            st.markdown(f"- {strength}")
                                    with col2:
                                        st.markdown("Áreas de mejora:")
                                        for weakness in analysis.get("weaknesses", []):
                                            st.markdown(f"- {weakness}")

                                    st.markdown("---")
                                    st.markdown("Resumen:")
                                    st.info(analysis.get("summary", "No disponible"))

                                    st.markdown("Contacto:")
                                    st.text(f"Email: {metadata.get('email', 'No disponible')}")

                            st.markdown("---")
                            if st.button("Generar Reporte Completo"):
                                report = generate_comparison_report(
                                    ideal_profile,
                                    filtered_candidates
                                )
                                st.text_area("Reporte de Comparación", report, height=400)

                except Exception as exc:
                    st.error(f"Error en la búsqueda: {exc}")
    
    with tab3:
        st.header("Ranking (Top N)")
        if not st.session_state.last_ranked_candidates:
            st.info("Primero ejecuta una búsqueda en la pestaña 'Buscar Candidatos'.")
        else:
            max_n = len(st.session_state.last_ranked_candidates)
            top_n = st.slider("Selecciona el Top N por match", 1, max_n, min(5, max_n))
            ranked = st.session_state.last_ranked_candidates[:top_n]

            for i, candidate in enumerate(ranked, 1):
                analysis = candidate.get("analysis", {})
                metadata = candidate.get("metadata", {})
                match_score = analysis.get("match_score", 0)
                vector_score = candidate.get("vector_score", 0)

                with st.expander(
                    f"#{i} {metadata.get('filename', 'Sin nombre')} - Match: {match_score}/100 (vector {vector_score:.3f})",
                    expanded=(i == 1),
                ):
                    st.markdown("Resumen:")
                    st.info(analysis.get("summary", "No disponible"))
                    st.markdown("Recomendación: " + analysis.get("recommendation", "N/A"))
                    st.markdown("Fortalezas:")
                    for strength in analysis.get("strengths", []):
                        st.markdown(f"- {strength}")
                    st.markdown("Áreas de mejora:")
                    for weakness in analysis.get("weaknesses", []):
                        st.markdown(f"- {weakness}")

            st.markdown("---")
            if st.button("Generar Reporte (Top N)"):
                report = generate_comparison_report(
                    st.session_state.last_ideal_profile,
                    ranked,
                )
                st.text_area("Reporte de Comparación", report, height=400)


if __name__ == "__main__":
    main()
