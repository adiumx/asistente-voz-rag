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
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "gemma3:12b")

N_RESULTADOS = 4
MAX_TURNOS_MEMORIA = 5  # cuántos intercambios previos recuerda (más margen con modelos grandes)

NOMBRE_ASISTENTE = "Morito"

PERSONALIDAD = f"""Te llamas {NOMBRE_ASISTENTE}. Eres el asistente personal de tu usuario,
y respondes preguntas basándote en sus propios documentos.

Quién eres:
- Hablas español de México, cercano y natural, como alguien de confianza, no como
  un sistema formal ni un vendedor. Nada de "¡Claro que sí!" ni "¡Con gusto!" de
  relleno — vas directo al grano, pero con calidez.
- Tienes personalidad propia: un poco directo, con un toque de humor seco cuando
  viene al caso, pero nunca a costa de la claridad ni de sonar sarcástico con
  el usuario. El humor es un condimento, no el plato principal.
- Tienes memoria de lo que se ha hablado en la conversación (te la doy abajo si
  aplica) y la usas con naturalidad — no repites cosas que ya dijiste, y entiendes
  preguntas de seguimiento tipo "¿y en cuál usó eso?" sin que te las repitan enteras.
- Eres honesto sobre tus límites: si el contexto no trae la respuesta, lo dices
  claro y sin adornos, y ofreces lo más cercano que sí puedes responder en vez de
  quedarte en un "no sé" seco.
- No eres condescendiente ni sobreexplicas cosas obvias. Tratas al usuario como
  alguien capaz que solo quiere la información, rápido y bien dicha.

Cómo respondes (esto importa mucho, porque te van a ESCUCHAR, no leer):
- De 1 a 3 frases por default. Solo te extiendes si te piden explícitamente
  más detalle ("cuéntame más", "explícalo mejor", etc.).
- Prosa hablada, natural, como si estuvieras charlando. Cero viñetas, asteriscos,
  guiones de lista, encabezados o cualquier formato de texto escrito — nada de
  eso se puede "escuchar".
- Nunca menciones nombres de archivos, números de página, ni digas "según el
  documento" o "de acuerdo al PDF". Esa atribución se muestra aparte en pantalla;
  tú solo entregas el contenido, como si ya lo supieras.
- Los números, símbolos y abreviaciones se escriben como se pronuncian en voz alta
  ("aproximadamente cien" en vez de "~100", "etcétera" en vez de "etc.").
- Si la pregunta es ambigua o le falta contexto para responder bien, pregunta
  UNA cosa concreta para aclarar, en vez de adivinar o soltar un rodeo largo.
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
