import gradio as gr
import numpy as np
import soundfile as sf
import whisper
import torch
import imageio_ffmpeg
import os
import re
import shutil
from kokoro import KPipeline

from consultar_rag import obtener_coleccion, preguntar

# --- Setup de ffmpeg portable (para Whisper) ---
ffmpeg_real_path = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_real_path)
ffmpeg_target = os.path.join(ffmpeg_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
if not os.path.exists(ffmpeg_target):
    shutil.copy(ffmpeg_real_path, ffmpeg_target)
os.environ["PATH"] += os.pathsep + ffmpeg_dir

# --- Setup de Whisper ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Whisper usando: {device}")
whisper_model = whisper.load_model("base", device=device)

# --- Setup de la base vectorial (RAG) ---
print("Cargando base de documentos...")
coleccion = obtener_coleccion()
print(f"Documentos indexados: {coleccion.count()} fragmentos")

# --- Setup de Kokoro TTS ---
print("Cargando modelo de voz (Kokoro)...")
tts_pipeline = KPipeline(lang_code='e')  # 'e' = español
VOZ_TTS = 'ef_dora'


def limpiar_texto_para_voz(texto):
    """Quita elementos que no deben leerse en voz alta."""
    texto_limpio = re.sub(r'\S+@\S+\.\S+', 'correo electronico', texto)
    texto_limpio = re.sub(r'\(Fuente:.*?\)', '', texto_limpio)
    texto_limpio = re.sub(r'\[Fuente:.*?\]', '', texto_limpio)
    texto_limpio = texto_limpio.replace('*', '').replace('•', '')
    texto_limpio = ' '.join(texto_limpio.split())
    return texto_limpio


def generar_audio_respuesta(texto):
    """Genera audio con Kokoro. Devuelve (samplerate, numpy_array) para Gradio."""
    texto_limpio = limpiar_texto_para_voz(texto)
    if not texto_limpio.strip():
        return None

    audio_completo = []
    for _, _, audio_chunk in tts_pipeline(texto_limpio, voice=VOZ_TTS):
        audio_completo.append(audio_chunk)

    if not audio_completo:
        return None

    audio_final = np.concatenate(audio_completo)
    return (24000, audio_final)


def procesar_pregunta(audio_grabado):
    """
    Recibe el audio grabado desde el navegador (Gradio), lo transcribe,
    busca en el RAG, genera la respuesta y la sintetiza en voz.
    """
    if audio_grabado is None:
        return "No se recibió audio.", "", None

    # Gradio entrega (samplerate, numpy_array)
    samplerate, audio_np = audio_grabado

    # Normaliza a float32 en el rango que espera Whisper
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32) / np.iinfo(audio_np.dtype).max \
            if np.issubdtype(audio_np.dtype, np.integer) else audio_np.astype(np.float32)

    # Si es estéreo, toma un solo canal
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    # Guarda temporalmente para pasarlo a Whisper
    ruta_temp = "temp_gradio.wav"
    sf.write(ruta_temp, audio_np, samplerate)

    # Transcribe
    resultado = whisper_model.transcribe(ruta_temp, language="es")
    pregunta = resultado["text"].strip()

    if not pregunta:
        return "No se entendió el audio.", "", None

    # Busca en el RAG y genera respuesta
    respuesta = preguntar(pregunta, coleccion, mostrar_fuentes=True)

    # Genera el audio de la respuesta
    audio_respuesta = generar_audio_respuesta(respuesta)

    return pregunta, respuesta, audio_respuesta


# --- Interfaz Gradio ---
with gr.Blocks(title="Asistente de Voz con RAG") as demo:
    gr.Markdown("# 🎙️ Asistente de Voz con RAG")
    gr.Markdown(
        "Graba tu pregunta sobre los documentos indexados. "
        "El asistente transcribe, busca en los PDFs, y responde con voz."
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Graba tu pregunta"
            )
            boton_enviar = gr.Button("Preguntar", variant="primary")

        with gr.Column():
            texto_pregunta = gr.Textbox(label="Transcripción de tu pregunta")
            texto_respuesta = gr.Textbox(label="Respuesta", lines=6)
            audio_output = gr.Audio(label="Respuesta en voz", autoplay=True)

    boton_enviar.click(
        fn=procesar_pregunta,
        inputs=audio_input,
        outputs=[texto_pregunta, texto_respuesta, audio_output]
    )

    gr.Markdown(f"*Documentos indexados: {coleccion.count()} fragmentos*")


if __name__ == "__main__":
    demo.launch()
