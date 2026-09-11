"""
Indexa archivos PDF y TXT en una base de datos vectorial (ChromaDB) para
usarlos como fuente de conocimiento en un pipeline RAG.

Uso:
    python indexar_pdfs.py                  # indexa todo lo soportado en ./documentos
    python indexar_pdfs.py ruta/a/mi.pdf     # indexa un archivo específico
    python indexar_pdfs.py ruta/a/mi.txt
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

EXTENSIONES_SOPORTADAS = (".pdf", ".txt")

CHUNK_SIZE = 1000        # caracteres por chunk
CHUNK_OVERLAP = 200      # solapamiento entre chunks consecutivos


def extraer_texto_pdf(ruta):
    """Extrae el texto de un PDF, página por página. Devuelve [(num_pagina, texto), ...]."""
    reader = PdfReader(ruta)
    paginas = []
    for i, page in enumerate(reader.pages):
        texto = page.extract_text() or ""
        if texto.strip():
            paginas.append((i + 1, texto))
    return paginas


def extraer_texto_txt(ruta):
    """Lee un .txt completo. Los .txt no tienen 'páginas', así que todo va en una sola."""
    # utf-8-sig quita el BOM si el archivo lo trae (común en archivos exportados de Windows)
    with open(ruta, "r", encoding="utf-8-sig", errors="replace") as f:
        texto = f.read()
    return [(1, texto)] if texto.strip() else []


def extraer_texto(ruta):
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".pdf":
        return extraer_texto_pdf(ruta)
    elif ext == ".txt":
        return extraer_texto_txt(ruta)
    else:
        raise ValueError(f"Formato no soportado: {ext}")


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


def indexar_documento(ruta, coleccion):
    nombre_archivo = os.path.basename(ruta)
    print(f"Procesando: {nombre_archivo}")

    try:
        paginas = extraer_texto(ruta)
    except ValueError as e:
        print(f"  -> Saltado: {e}")
        return

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

    unidad = "páginas" if ruta.lower().endswith(".pdf") else "bloques"
    print(f"  -> {len(paginas)} {unidad}, {total_chunks} fragmentos indexados")


def main():
    os.makedirs(CARPETA_DOCS, exist_ok=True)

    client = chromadb.PersistentClient(path=CARPETA_DB)
    coleccion = client.get_or_create_collection(COLECCION)

    # Argumento opcional: un archivo específico
    if len(sys.argv) > 1:
        archivos = [sys.argv[1]]
    else:
        archivos = [
            os.path.join(CARPETA_DOCS, f)
            for f in os.listdir(CARPETA_DOCS)
            if f.lower().endswith(EXTENSIONES_SOPORTADAS)
        ]

    if not archivos:
        print(f"No se encontraron PDF ni TXT en {CARPETA_DOCS}")
        print("Coloca tus archivos .pdf o .txt ahí, o pasa la ruta como argumento.")
        return

    for ruta in archivos:
        indexar_documento(ruta, coleccion)

    print(f"\nListo. Total de fragmentos en la colección: {coleccion.count()}")


if __name__ == "__main__":
    main()
