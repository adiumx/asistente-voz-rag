# Asistente de Voz con RAG (Español)

Asistente de voz local que responde preguntas basándose en tus propios documentos PDF. Todo corre 100% en local: sin APIs externas, sin enviar datos a la nube.

<!-- Reemplaza esto con tu video/GIF cuando lo tengas -->
<!-- ![Demo](docs/demo.gif) -->
[Ver video demo](https://youtube.com/shorts/_QPBMXimtCM?feature=share)

## Cómo funciona

```
🎙️ Voz  →  📝 Whisper (transcripción)  →  🔍 Búsqueda semántica en tus PDFs
                                                    ↓
🔊 Kokoro TTS  ←  💬 Respuesta hablada  ←  🧠 Gemma 3 (genera la respuesta)
```

1. **Push-to-talk**: presionas una tecla, hablas tu pregunta
2. **Whisper** transcribe tu voz a texto
3. El sistema busca los fragmentos más relevantes de tus PDFs indexados (embeddings + búsqueda vectorial con ChromaDB)
4. **Gemma 3** genera una respuesta coherente basada únicamente en esos fragmentos
5. **Kokoro TTS** convierte la respuesta a voz y la reproduce

## Stack técnico

| Componente | Tecnología |
|---|---|
| Transcripción de voz | [OpenAI Whisper](https://github.com/openai/whisper) (local) |
| Detección de silencio | WebRTC VAD |
| LLM generativo | [Gemma 3](https://ollama.com/library/gemma3) vía [Ollama](https://ollama.com) |
| Embeddings | nomic-embed-text |
| Base vectorial | [ChromaDB](https://www.trychroma.com/) |
| Extracción de PDF | pypdf |
| Texto a voz | [Kokoro TTS](https://github.com/hexgrad/kokoro) |

## Instalación

Requiere Linux con GPU NVIDIA (probado en RTX 3060 / Tesla V100).

```bash
git clone https://github.com/TU_USUARIO/TU_REPO.git
cd TU_REPO
make install
```

Esto instala las dependencias del sistema, crea el entorno virtual de Python, y descarga los modelos necesarios (Gemma 3, nomic-embed-text).

## Uso

```bash
# 1. Coloca tus PDFs en la carpeta documentos/
cp mi_archivo.pdf documentos/

# 2. Indexa los documentos
make index

# 3. Corre el asistente
make run
```

Presiona `Ctrl+Shift+A` para hablar, espera a que detecte silencio, y escucha la respuesta.

## Estructura del proyecto

```
.
├── indexar_pdfs.py       # Extrae texto de PDFs y los indexa en ChromaDB
├── consultar_rag.py      # Búsqueda semántica + generación de respuesta
├── main_rag_kokoro.py    # Loop principal: voz → RAG → voz
├── requirements.txt
├── Makefile
└── documentos/           # Coloca aquí tus PDFs
```

## Por qué estas decisiones técnicas

- **Todo local**: privacidad de los documentos, sin costos de API, sin depender de conexión a internet.
- **Push-to-talk en vez de wake word**: se experimentó con un modelo de wake word personalizado en español (entrenado con [openWakeWord](https://github.com/dscripka/openWakeWord)), pero con los datos sintéticos disponibles el modelo generaba demasiados falsos positivos para uso práctico. Push-to-talk resultó más confiable para este caso de uso.
- **Kokoro sobre Piper**: mejor calidad de voz y manejo más natural del texto en español (tildes, símbolos), con un costo de recursos similar.

## Licencia

MIT
