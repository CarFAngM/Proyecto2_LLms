# Proyecto2_LLms
 
Este proyecto consiste en un asistente inteligente capaz de responder preguntas relacionadas con áreas de interes y otras disciplinas académicas. Utiliza tecnología de IA generativa (OpenAI), búsqueda semántica con vectores y almacenamiento en Pinecone, y una interfaz interactiva desarrollada con Streamlit. los documentos se tratan de economia, calculo y demas

Instrucciones para la ejecución

Clonar el repositorio


git clone https://github.com/tu_usuario/asistente-estudiantil.git
cd asistente-estudiantil

Crear un entorno virtual

python -m venv venv
source venv/bin/activate     
venv\Scripts\activate


Instalar dependencias
pip install -r requirements.txt

Crear archivo .env con las claves necesarias:
OPENAI_API_KEY=tu_clave_de_openai
PINECONE_API_KEY=tu_clave_de_pinecone
INDEX_NAME=proyecto-index2

Ejecutar la aplicación

streamlit run main.py

Descripción de los módulos del proyecto

main.py
Archivo principal que lanza la aplicación. Configura la interfaz con Streamlit, carga los documentos al índice de Pinecone, y ejecuta el motor de búsqueda con inteligencia artificial.

modulos/pinecone_c.py
Contiene la función init_pinecone() para inicializar la conexión con Pinecone utilizando la nueva clase Pinecone correctamente.

modulos/cargar_docs.py
Contiene funciones para leer archivos desde una carpeta, generar los embeddings con OpenAI y cargarlos en Pinecone, manejando fragmentación de texto y errores.

docs/
Carpeta que contiene los documentos base desde los cuales se puede responder con el asistente. Deben estar en formato .pdf, .txt o .md.

.env
Archivo oculto con las claves API necesarias para OpenAI y Pinecone. No debe compartirse ni subirse al repositorio por seguridad.

requirements.txt
Lista de dependencias necesarias para que el entorno de Python ejecute correctamente el proyecto.

¿Qué aprendimos al desarrollar el asistente?

Aprendizajes técnicos

Cómo usar modelos de lenguaje como GPT para tareas de búsqueda aumentada por recuperación.

Cómo usar embeddings para representar documentos y almacenarlos en Pinecone.

Cómo integrar múltiples servicios externos (OpenAI, Pinecone) de forma segura.


Aprendizajes en equipo o personales

La importancia de dividir el proyecto en módulos reutilizables.

Coordinación y comunicación para que cada miembro trabaje en partes independientes pero compatibles.

Mejores prácticas para manejar errores, entorno virtual, y gestión de secretos con .env.