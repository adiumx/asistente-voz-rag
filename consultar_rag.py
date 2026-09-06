"""
Búsqueda semántica sobre los PDFs indexados + generación de respuesta con Gemma.

Cambios respecto a la versión anterior:
- `preguntar()` ahora devuelve (texto, fuentes) por separado, para que la voz
  nunca lea las fuentes pero la UI sí las pueda mostrar.
- Prompt con personalidad y respuestas cortas (pensadas para ser habladas).
- Memoria conversacional corta, para poder hacer preguntas de seguimiento.
"""

import os
import chromadb
import ollama

CARPETA_DB = "./chroma_db"
COLECCION = "pdfs"
EMBED_MODEL = "nomic-embed-text"

# Configurable por entorno: en la 3060 usa gemma3:4b, en la V100 gemma3:12b
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "gemma3:4b")

N_RESULTADOS = 4
MAX_TURNOS_MEMORIA = 3  # cuántos intercambios previos recuerda

NOMBRE_ASISTENTE = "Morito"

PERSONALIDAD = f"""Te llamas {NOMBRE_ASISTENTE}. Eres un asistente de voz que responde
preguntas sobre los documentos de la persona con la que hablas.

Cómo eres:
- Cercano y directo, con un tono relajado pero profesional. Hablas español de México.
- Breve: tus respuestas se escuchan en voz alta, así que van de 1 a 3 frases.
  Solo te extiendes si te piden explícitamente más detalle.
- Nunca inventas. Si algo no está en el contexto, lo dices sin rodeos y ofreces
  lo que sí puedes responder.
- Ocasionalmente tienes un toque de humor seco, pero nunca a costa de la claridad.

Reglas de formato (importantes, porque te van a leer en voz alta):
- Escribe en prosa natural. Nada de viñetas, asteriscos, guiones de lista ni markdown.
- No menciones nombres de archivos, páginas ni "según el documento". Esa parte se
  muestra aparte en pantalla. Tú solo responde el contenido.
- Escribe los números y símbolos como se pronuncian (por ejemplo "aproximadamente cien"
  en lugar de "~100").
"""


class MotorRAG:
    """Encapsula la colección y la memoria de la conversación."""

    def __init__(self, carpeta_db=CARPETA_DB, coleccion=COLECCION):
        self.client = chromadb.PersistentClient(path=carpeta_db)
        self.coleccion = self.client.get_or_create_collection(coleccion)
        self.historial = []  # [(pregunta, respuesta), ...]

    def contar(self):
        return self.coleccion.count()

    def limpiar_memoria(self):
        self.historial = []

    def buscar_contexto(self, pregunta, n=N_RESULTADOS):
        emb = ollama.embeddings(model=EMBED_MODEL, prompt=pregunta)["embedding"]
        res = self.coleccion.query(query_embeddings=[emb], n_results=n)

        if not res["documents"] or not res["documents"][0]:
            return []

        fragmentos = []
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            fragmentos.append({
                "texto": doc,
                "archivo": meta.get("archivo", "desconocido"),
                "pagina": meta.get("pagina", "?"),
            })
        return fragmentos

    def _construir_prompt(self, pregunta, fragmentos):
        contexto = "\n\n".join(
            f"--- Fragmento {i+1} ---\n{f['texto']}"
            for i, f in enumerate(fragmentos)
        )

        memoria = ""
        if self.historial:
            turnos = self.historial[-MAX_TURNOS_MEMORIA:]
            memoria = "\nConversación reciente (para entender preguntas de seguimiento):\n"
            memoria += "\n".join(f"Persona: {p}\nTú: {r}" for p, r in turnos)
            memoria += "\n"

        return f"""{PERSONALIDAD}
{memoria}
Contexto extraído de los documentos:
{contexto}

Pregunta: {pregunta}

Tu respuesta (1 a 3 frases, en prosa, sin mencionar archivos ni páginas):"""

    def preguntar(self, pregunta):
        """
        Devuelve (texto_respuesta, lista_de_fuentes).
        `fuentes` es una lista de dicts {archivo, pagina} sin duplicados.
        """
        fragmentos = self.buscar_contexto(pregunta)

        if not fragmentos:
            texto = ("No encontré nada sobre eso en tus documentos. "
                     "¿Quieres preguntarme otra cosa?")
            self.historial.append((pregunta, texto))
            return texto, []

        prompt = self._construir_prompt(pregunta, fragmentos)

        try:
            respuesta = ollama.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={"temperature": 0.6, "num_predict": 220},
            )
            texto = respuesta["response"].strip()
        except Exception as e:
            return f"No pude generar la respuesta. Error: {e}", []

        # Fuentes únicas, preservando el orden de relevancia
        vistas = set()
        fuentes = []
        for f in fragmentos:
            clave = (f["archivo"], f["pagina"])
            if clave not in vistas:
                vistas.add(clave)
                fuentes.append({"archivo": f["archivo"], "pagina": f["pagina"]})

        self.historial.append((pregunta, texto))
        return texto, fuentes


# --- Compatibilidad con los scripts anteriores ---

_motor_global = None


def obtener_coleccion():
    global _motor_global
    if _motor_global is None:
        _motor_global = MotorRAG()
    return _motor_global


def preguntar(pregunta, motor=None, mostrar_fuentes=False):
    motor = motor or obtener_coleccion()
    texto, fuentes = motor.preguntar(pregunta)
    if mostrar_fuentes and fuentes:
        etiquetas = ", ".join(f"{f['archivo']} (p. {f['pagina']})" for f in fuentes)
        texto += f"\n\nFuentes: {etiquetas}"
    return texto


if __name__ == "__main__":
    motor = MotorRAG()
    print(f"Documentos indexados: {motor.contar()} fragmentos\n")
    print("Escribe tu pregunta (o 'salir')\n")

    while True:
        try:
            pregunta = input("Pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pregunta.lower() in ("salir", "exit", "quit"):
            break
        if not pregunta:
            continue

        texto, fuentes = motor.preguntar(pregunta)
        print(f"\n{NOMBRE_ASISTENTE}: {texto}")
        if fuentes:
            etiquetas = " · ".join(f"{f['archivo']} p.{f['pagina']}" for f in fuentes)
            print(f"  [{etiquetas}]")
        print()
