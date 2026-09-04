"""
Indexa archivos PDF en una base de datos vectorial (ChromaDB) para usarlos
como fuente de conocimiento en un pipeline RAG.

Uso:
    python indexar_pdfs.py                  # indexa todos los PDF en ./documentos
    python indexar_pdfs.py ruta/a/mi.pdf     # indexa un PDF específico
"""

import os
import sys
import hashlib
import chromadb
import ollama
from pypdf import PdfReader

CARPETA_DOCS = "./documentos"
CARPETA_DB = "./chroma_db"
COLECCION = "pdfs"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1000        # caracteres por chunk
CHUNK_OVERLAP = 200      # solapamiento entre chunks consecutivos


def extraer_texto_pdf(ruta_pdf):
    """Extrae todo el texto de un PDF, página por página."""
    reader = PdfReader(ruta_pdf)
    paginas = []
    for i, page in enumerate(reader.pages):
        texto = page.extract_text() or ""
        if texto.strip():
            paginas.append((i + 1, texto))
    return paginas


def trocear_texto(texto, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Divide un texto largo en fragmentos con solapamiento."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + chunk_size
        chunk = texto[inicio:fin]
        if chunk.strip():
            chunks.append(chunk)
        inicio += chunk_size - overlap
    return chunks


def generar_id(archivo, pagina, chunk_idx):
    """Genera un ID único y determinista para cada chunk."""
    base = f"{archivo}-{pagina}-{chunk_idx}"
    return hashlib.md5(base.encode()).hexdigest()


def indexar_pdf(ruta_pdf, coleccion):
    nombre_archivo = os.path.basename(ruta_pdf)
    print(f"Procesando: {nombre_archivo}")

    paginas = extraer_texto_pdf(ruta_pdf)
    total_chunks = 0

    for num_pagina, texto_pagina in paginas:
        chunks = trocear_texto(texto_pagina)
        for idx, chunk in enumerate(chunks):
            chunk_id = generar_id(nombre_archivo, num_pagina, idx)

            # Genera el embedding usando el modelo local vía Ollama
            respuesta = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
            embedding = respuesta["embedding"]

            coleccion.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "archivo": nombre_archivo,
                    "pagina": num_pagina,
                }]
            )
            total_chunks += 1

    print(f"  -> {len(paginas)} páginas, {total_chunks} fragmentos indexados")


def main():
    os.makedirs(CARPETA_DOCS, exist_ok=True)

    client = chromadb.PersistentClient(path=CARPETA_DB)
    coleccion = client.get_or_create_collection(COLECCION)

    # Argumento opcional: un PDF específico
    if len(sys.argv) > 1:
        archivos = [sys.argv[1]]
    else:
        archivos = [
            os.path.join(CARPETA_DOCS, f)
            for f in os.listdir(CARPETA_DOCS)
            if f.lower().endswith(".pdf")
        ]

    if not archivos:
        print(f"No se encontraron PDFs en {CARPETA_DOCS}")
        print("Coloca tus archivos .pdf ahí, o pasa la ruta como argumento.")
        return

    for ruta in archivos:
        indexar_pdf(ruta, coleccion)

    print(f"\nListo. Total de documentos en la colección: {coleccion.count()}")


if __name__ == "__main__":
    main()
