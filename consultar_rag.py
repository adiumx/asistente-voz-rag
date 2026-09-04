"""
Consulta la base vectorial de PDFs y genera una respuesta coherente
usando Gemma 3 (vía Ollama), citando de qué documento/página viene
la información.
"""

import chromadb
import ollama

CARPETA_DB = "./chroma_db"
COLECCION = "pdfs"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:4b" #LLM_MODEL = "gemma3:12b"
N_RESULTADOS = 4  # cuántos fragmentos relevantes recuperar por pregunta


def obtener_coleccion():
    client = chromadb.PersistentClient(path=CARPETA_DB)
    return client.get_or_create_collection(COLECCION)


def buscar_contexto(pregunta, coleccion, n=N_RESULTADOS):
    """Busca los fragmentos más relevantes para la pregunta."""
    respuesta = ollama.embeddings(model=EMBED_MODEL, prompt=pregunta)
    embedding_pregunta = respuesta["embedding"]

    resultados = coleccion.query(
        query_embeddings=[embedding_pregunta],
        n_results=n
    )

    fragmentos = []
    for doc, meta in zip(resultados["documents"][0], resultados["metadatas"][0]):
        fragmentos.append({
            "texto": doc,
            "archivo": meta["archivo"],
            "pagina": meta["pagina"],
        })
    return fragmentos


def construir_prompt(pregunta, fragmentos):
    contexto = "\n\n".join(
        f"[Fuente: {f['archivo']}, página {f['pagina']}]\n{f['texto']}"
        for f in fragmentos
    )

    prompt = f"""Eres un asistente que responde preguntas basándote ÚNICAMENTE en el contexto proporcionado.
Si la respuesta no está en el contexto, di claramente que no tienes esa información en los documentos.
Responde en español, de forma clara y directa, en prosa natural sin usar viñetas, asteriscos, ni símbolos especiales.
Cuando sea relevante, menciona de qué documento sacaste la información de forma natural dentro del texto, no entre paréntesis.

Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:"""
    return prompt


def preguntar(pregunta, coleccion=None, mostrar_fuentes=True):
    if coleccion is None:
        coleccion = obtener_coleccion()

    fragmentos = buscar_contexto(pregunta, coleccion)

    if not fragmentos:
        return "No encontré información relevante en los documentos indexados."

    prompt = construir_prompt(pregunta, fragmentos)

    respuesta = ollama.generate(model=LLM_MODEL, prompt=prompt)
    texto_respuesta = respuesta["response"].strip()

    if mostrar_fuentes:
        fuentes = sorted(set(f"{f['archivo']} (pág. {f['pagina']})" for f in fragmentos))
        texto_respuesta += "\n\nFuentes: " + ", ".join(fuentes)

    return texto_respuesta


if __name__ == "__main__":
    coleccion = obtener_coleccion()
    print(f"Documentos indexados: {coleccion.count()} fragmentos\n")

    print("Escribe tu pregunta (o 'salir' para terminar)\n")
    while True:
        pregunta = input("Pregunta: ").strip()
        if pregunta.lower() in ("salir", "exit", "quit"):
            break
        if not pregunta:
            continue

        respuesta = preguntar(pregunta, coleccion)
        print(f"\nRespuesta: {respuesta}\n")
