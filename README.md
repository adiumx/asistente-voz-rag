# 🎙️ Morito — Asistente de Voz con RAG (100% local)

Asistente de voz con inteligencia artificial que responde preguntas basándose en tus propios documentos (PDF y TXT). Todo corre localmente en tu hardware: sin APIs externas, sin enviar datos a la nube, con una interfaz gráfica flotante siempre disponible.

[▶ Ver video demo](https://youtube.com/shorts/_QPBMXimtCM?feature=share)

---

## En resumen (para quien tiene 30 segundos)

Un asistente tipo "Siri/Alexa" pero **privado**: vive como un pequeño orbe flotante en tu escritorio, escucha tu pregunta, busca la respuesta en tus propios documentos, y te contesta en voz — todo procesado en tu propia máquina, sin que ni un byte salga a internet. Tiene personalidad propia, recuerda el contexto de la conversación, y responde en menos de 4 segundos gracias a una arquitectura optimizada para GPU.

| | |
|---|---|
| 🔒 **Privacidad** | Cero llamadas a APIs externas — documentos, voz y respuestas nunca salen de tu equipo |
| ⚡ **Rápido** | 3.85s de latencia hasta la primera respuesta (bajado desde 43s — ver sección de rendimiento) |
| 🎭 **Con personalidad** | No es un chatbot genérico: tiene nombre, tono y estilo conversacional propios |
| 🖥️ **Interfaz nativa** | Orbe flotante animado, siempre accesible, sin depender de una terminal |
| 🧩 **Multiplataforma** | Windows y Linux |

---

## Cómo funciona

```
                    🎙️ Presionas la tecla y hablas
                              │
                              ▼
                    📝 Whisper transcribe tu voz
                              │
                              ▼
              🔍 Busca los fragmentos más relevantes
                 de tus documentos (ChromaDB)
                              │
                              ▼
              🧠 Gemma 3 genera una respuesta, con
              personalidad y memoria de la conversación
                              │
                              ▼
              🔊 Kokoro convierte la respuesta a voz,
              hablando cada frase apenas está lista
                 (no espera a tener todo el audio)
                              │
                              ▼
        Ves la respuesta y las fuentes en una tarjeta
        discreta junto al orbe — nunca se leen en voz alta
```

**El orbe flotante** cambia de color y anima según lo que está haciendo: escuchando, pensando, respondiendo. Un clic mientras habla lo interrumpe al instante. Clic derecho abre un menú para silenciar la voz o reindexar documentos sin tocar la terminal.

---

## Rendimiento: de 43 segundos a 3.85 segundos

Durante el desarrollo, el mayor cuello de botella no era el reconocimiento de voz ni el modelo de lenguaje — era la síntesis de voz corriendo en CPU en lugar de GPU. Medido con instrumentación real (no estimaciones):

| Etapa | Antes | Después | Mejora |
|---|---|---|---|
| Síntesis de voz (Kokoro) | 10.99s | 1.16s | **9.5× más rápido** |
| Latencia hasta la primera respuesta audible | 14.24s | 3.85s | **3.7× más rápido** |

El diagnóstico se hizo instrumentando cada etapa del pipeline con timestamps reales, no adivinando — un patrón de trabajo aplicado en varios otros problemas del proyecto (ver sección de ingeniería abajo).

---

## Stack técnico

| Componente | Tecnología | Por qué |
|---|---|---|
| Transcripción de voz | [OpenAI Whisper](https://github.com/openai/whisper) | Robusto en español, corre local en GPU |
| Reducción de ruido | [noisereduce](https://github.com/timsainb/noisereduce) | Filtra ruido constante de fondo (ventilador de GPU) antes de transcribir |
| Detección de silencio | WebRTC VAD | Detecta cuándo terminaste de hablar sin depender de temporizadores fijos |
| LLM generativo | [Gemma 3](https://ollama.com/library/gemma3) vía [Ollama](https://ollama.com) | Modelo local, sin costos de API por consulta |
| Embeddings | nomic-embed-text | Búsqueda semántica, no solo coincidencia de palabras |
| Base vectorial | [ChromaDB](https://www.trychroma.com/) | Persistente, ligera, sin servidor externo |
| Texto a voz | [Kokoro TTS](https://github.com/hexgrad/kokoro) | Calidad natural con bajo costo de cómputo (82M parámetros) |
| Interfaz gráfica | [PySide6](https://doc.qt.io/qtforpython/) | Ventana flotante nativa, siempre encima, sin marco |
| Concurrencia | `multiprocessing` + `QThread` | Ver "Arquitectura" abajo — es la pieza que hace todo esto estable |

---

## Arquitectura: por qué dos procesos, no uno

La primera versión corría todo (interfaz gráfica, transcripción, generación, síntesis) en un solo programa con hilos. En Windows, esto producía **cierres inesperados sin ningún mensaje de error** cada vez que el modelo de IA hacía su primer cálculo real en un hilo secundario — un problema de bajo nivel entre PyTorch y el sistema de hilos de Windows, no capturable con `try/except` normal.

La solución: separar el cómputo pesado (Whisper, búsqueda RAG, Kokoro) en un **proceso independiente**, que se comunica con la interfaz gráfica mediante colas de mensajes.

```
┌─────────────────────────┐        Colas de mensajes        ┌──────────────────────────┐
│   Proceso de interfaz    │ ───────────────────────────────▶│   Proceso de inferencia   │
│                          │                                  │                          │
│  • Orbe flotante (Qt)    │ ◀─────────────────────────────── │  • Whisper                │
│  • Grabación de audio    │      Texto, audio, estado        │  • Búsqueda RAG           │
│  • Nunca se congela      │                                  │  • Gemma 3                │
└─────────────────────────┘                                  │  • Kokoro                 │
                                                               └──────────────────────────┘
```

Si el proceso de inferencia llegara a fallar, la interfaz lo detecta y lo **reinicia automáticamente** — el usuario nunca ve una ventana congelada o un cierre silencioso.

---

## Instalación

Requiere Linux con GPU NVIDIA (probado en RTX 3060 y Tesla V100) o Windows.

```bash
git clone https://github.com/TU_USUARIO/TU_REPO.git
cd TU_REPO
make install
```

Esto instala las dependencias del sistema, crea el entorno virtual de Python, instala PyTorch con soporte CUDA, y descarga los modelos de Ollama (Gemma 3, nomic-embed-text).

## Uso

```bash
# 1. Coloca tus documentos (PDF o TXT) en la carpeta documentos/
cp mi_archivo.pdf documentos/

# 2. Indexa
make index

# 3. Corre el asistente
make run
```

El orbe aparece en la esquina de la pantalla. Presiona la combinación de teclas configurada (o haz clic en el orbe) para hablar.

```bash
make help   # ver todos los comandos disponibles
```

---

## Estructura del proyecto

```
.
├── asistente.py          # Interfaz gráfica (orbe flotante) + grabación de audio
├── motor_inferencia.py   # Proceso de inferencia: Whisper + RAG + Kokoro
├── orbe.py                # Widgets Qt: el orbe animado y la tarjeta de respuesta
├── consultar_rag.py       # Búsqueda semántica + personalidad del asistente
├── indexar_pdfs.py        # Indexa PDF y TXT en la base vectorial
├── requirements.txt
├── Makefile
└── documentos/             # Coloca aquí tus PDF/TXT
```

---

## Decisiones de ingeniería (y por qué se tomaron)

Esta sección documenta problemas reales encontrados durante el desarrollo — el tipo de troubleshooting que no se ve en un demo, pero es donde realmente pasa el trabajo de ingeniería.

**Wake word personalizado, descartado a favor de push-to-talk.**
Se entrenó un modelo de activación por voz en español ("oye morito") con [openWakeWord](https://github.com/dscripka/openWakeWord), generando datos sintéticos con Piper TTS y mezclándolos con ruido de fondo real (AudioSet, grabaciones propias). El modelo resultante tenía una tasa de falsos positivos demasiado alta para uso práctico — la causa raíz: el proyecto original usa ~900 voces y 2,000 horas de audio real como negativos, una escala imposible de replicar con recursos personales. En vez de perseguir una solución sobre-ingenierizada, se optó por push-to-talk: más simple, 100% confiable, y no le resta valor al producto final.

**Crashes silenciosos de PyTorch en Windows, resueltos con aislamiento de procesos.**
Documentado en la sección de arquitectura arriba — diagnosticado metódicamente probando CPU vs GPU, con/sin subprocess de ffmpeg, y limitando threads de OpenMP, hasta aislar la causa exacta y aplicar la solución estructural correcta (proceso separado) en vez de parches temporales.

**Optimización de latencia guiada por datos, no por intuición.**
En vez de asumir dónde estaba el cuello de botella, se instrumentó cada etapa del pipeline con mediciones reales. Los datos mostraron que Kokoro corriendo en CPU (decisión inicial para "ahorrar VRAM") era responsable del 75% del tiempo de respuesta — con margen de sobra en la GPU, moverlo a CUDA dio una mejora de 9.5× sin ningún otro cambio.

**Calibración de audio contra ruido de hardware real.**
El ventilador de una Tesla V100 (diseñada para servidores, no para silencio) interfería tanto con la detección de silencio como con la transcripción. Se resolvió midiendo el nivel real de ruido vs. voz con el micrófono específico del usuario, en vez de usar un umbral genérico — y agregando reducción de ruido espectral antes de transcribir.

**Streaming de audio para reducir la latencia percibida.**
Kokoro genera el audio frase por frase internamente. En vez de esperar a tener la respuesta completa sintetizada antes de reproducir nada, cada fragmento se envía y reproduce en cuanto está listo — el usuario empieza a escuchar la respuesta mientras el resto todavía se está generando.

---

## Licencia

MIT
