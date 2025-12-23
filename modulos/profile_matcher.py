"""
Comparación de perfiles ideales con CVs en modo funcional.
"""

import os
import json
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def _get_client(api_key: str = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Se requiere OPENAI_API_KEY")
    return OpenAI(api_key=key)


def analyze_candidate(ideal_profile: str, candidate_metadata: Dict, api_key: str = None) -> Dict:
    try:
        client = _get_client(api_key)
        filename = candidate_metadata.get("filename", "desconocido")
        email = candidate_metadata.get("email", "no disponible")
        word_count = candidate_metadata.get("word_count", 0)
        cv_text = candidate_metadata.get("cv_text", "")

        if not cv_text or len(cv_text.strip()) == 0:
            cv_info = f"""
            - Nombre de archivo: {filename}
            - Email: {email}
            - Palabras en CV: {word_count}

            NOTA: El texto del CV no está disponible en la metadata. Análisis basado solo en información limitada."""
        else:
            cv_info = f"""
            - Nombre de archivo: {filename}
            - Email: {email}
            - Palabras en CV: {word_count}

            CONTENIDO DEL CV (primeros 3000 caracteres):
            {cv_text[:3000]}
            """

        prompt = f"""Eres un reclutador EXTREMADAMENTE exigente y crítico. Tu trabajo es encontrar SOLO los mejores candidatos.

PERFIL IDEAL REQUERIDO:
{ideal_profile}

INFORMACIÓN DEL CANDIDATO:
{cv_info}

INSTRUCCIONES CRÍTICAS PARA LA EVALUACIÓN:

1. **ESCALA DE PUNTUACIÓN ESTRICTA:**
   - 90-100: Candidato EXCEPCIONAL. Cumple TODO y excede expectativas. Muy raro.
   - 75-89: Candidato EXCELENTE. Cumple todas las habilidades principales + experiencia requerida.
   - 60-74: Candidato BUENO. Cumple mayoría de requisitos pero le faltan 1-2 habilidades importantes.
   - 40-59: Candidato REGULAR. Cumple solo algunos requisitos. Faltan habilidades críticas.
   - 20-39: Candidato DÉBIL. Perfil muy diferente. Pocas coincidencias.
   - 0-19: Candidato INADECUADO. Rol completamente diferente o sin experiencia relevante.

2. **PENALIZACIONES OBLIGATORIAS:**
   - Si el ROL/TÍTULO es diferente (ej: RRHH vs Data Scientist): máximo 25 puntos
   - Si falta la tecnología PRINCIPAL del puesto: -20 puntos mínimo
   - Si la experiencia es menor a la requerida: -15 puntos mínimo
   - Si faltan 3+ habilidades técnicas requeridas: máximo 40 puntos
   - Si el área de especialización es diferente: máximo 30 puntos

3. **REQUISITOS PARA PUNTUACIONES ALTAS:**
   - 80+ puntos: DEBE tener el mismo rol/título Y todas las tecnologías principales
   - 70+ puntos: DEBE tener experiencia en el área específica requerida
   - 60+ puntos: DEBE cumplir al menos 70% de los requisitos técnicos

4. **SÉ MUY ESPECÍFICO:** No des puntos por "habilidades transferibles" vagas. Si pide Python y el candidato solo tiene Java, eso es 0 puntos en Python, no 5.

Analiza el CV y sé BRUTAL en tu evaluación. Es mejor rechazar un candidato mediocre que aprobar uno inadecuado.

Responde en formato JSON:
{{
  "match_score": [número 0-100 siguiendo la escala ESTRICTA],
  "strengths": ["Solo fortalezas que REALMENTE aplican al perfil buscado"],
  "weaknesses": ["TODAS las habilidades críticas faltantes", "Diferencias en rol/experiencia"],
  "summary": "Evaluación honesta y crítica del candidato",
  "recommendation": "[Highly Recommended|Recommended|Consider|Not Recommended]"
}}

RECUERDA: Ser exigente NO es ser injusto. Es asegurar que solo los mejores candidatos pasen el filtro.
"""

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un reclutador EXTREMADAMENTE exigente y crítico. Solo apruebas candidatos que realmente cumplen los requisitos. Sé honesto y estricto en tus evaluaciones. Responde SOLO con JSON válido.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        analysis = json.loads(content)

        required_fields = ["match_score", "strengths", "weaknesses", "summary", "recommendation"]
        for field in required_fields:
            if field not in analysis:
                raise ValueError(f"Campo requerido faltante: {field}")

        match_score = analysis.get("match_score", 0)
        recommendation = analysis.get("recommendation", "")

        if match_score >= 85 and recommendation not in ["Highly Recommended"]:
            analysis["recommendation"] = "Highly Recommended"
        elif match_score >= 70 and recommendation == "Not Recommended":
            analysis["recommendation"] = "Recommended"
        elif match_score < 40 and recommendation in ["Highly Recommended", "Recommended"]:
            analysis["recommendation"] = "Not Recommended"
        elif match_score < 60 and recommendation == "Highly Recommended":
            analysis["recommendation"] = "Consider"

        return analysis
    except json.JSONDecodeError as exc:
        return {
            "match_score": 0,
            "strengths": ["No se pudo analizar debido a error de formato"],
            "weaknesses": ["Error en decodificación JSON", "Análisis incompleto"],
            "summary": f"Error al procesar la respuesta del análisis: {exc}",
            "recommendation": "Not Recommended",
        }
    except Exception as exc:
        return {
            "match_score": 0,
            "strengths": ["Error al analizar candidato"],
            "weaknesses": ["No se pudo completar el análisis", "Error del sistema"],
            "summary": f"Se produjo un error durante el análisis: {exc}",
            "recommendation": "Not Recommended",
        }


def rank_candidates(ideal_profile: str, candidates: List[Dict], top_k: int, progress_callback=None, api_key: str = None) -> List[Dict]:
    top_candidates = candidates[:top_k]
    results: List[Dict] = []

    for i, candidate in enumerate(top_candidates, 1):
        analysis = analyze_candidate(ideal_profile, candidate.get("metadata", {}), api_key=api_key)
        results.append({
            "id": candidate.get("id"),
            "vector_score": candidate.get("score", 0),
            "metadata": candidate.get("metadata", {}),
            "analysis": analysis,
            "page_matches": candidate.get("page_matches", []),
        })
        if progress_callback:
            progress_callback(i, len(top_candidates))

    results.sort(key=lambda x: x["analysis"].get("match_score", 0), reverse=True)
    return results


def generate_comparison_report(ideal_profile: str, ranked_candidates: List[Dict]) -> str:
    report = f"""
╔══════════════════════════════════════════════════════════╗
║          REPORTE DE COMPARACIÓN DE CANDIDATOS            ║
╚══════════════════════════════════════════════════════════╝

PERFIL IDEAL:
{ideal_profile}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CANDIDATOS RANKEADOS:

"""

    for i, candidate in enumerate(ranked_candidates, 1):
        analysis = candidate.get("analysis", {})
        metadata = candidate.get("metadata", {})

        report += f"""
┌─ CANDIDATO #{i} ─────────────────────────────────────────
│ Archivo: {metadata.get('filename', 'N/A')}
│ Email: {metadata.get('email', 'N/A')}
│ Score de Coincidencia: {analysis.get('match_score', 0)}/100
│ Score Vectorial Promedio: {candidate.get('vector_score', 0):.3f}
│ Recomendación: {analysis.get('recommendation', 'N/A')}
│
│ FORTALEZAS:
"""
        for strength in analysis.get("strengths", []):
            report += f"│   ✓ {strength}\n"

        report += f"│\n│ ÁREAS DE MEJORA:\n"
        for weakness in analysis.get("weaknesses", []):
            report += f"│   ✗ {weakness}\n"

        report += f"│\n│ RESUMEN:\n│   {analysis.get('summary', 'N/A')}\n"
        report += "└─────────────────────────────────────────────────────────\n"

    report += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    report += f"Total de candidatos analizados: {len(ranked_candidates)}\n"
    return report
