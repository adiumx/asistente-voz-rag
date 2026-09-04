import webrtcvad
import sounddevice as sd
import numpy as np
import soundfile as sf
import whisper
import torch
import imageio_ffmpeg
import os
import re
import shutil
import keyboard
from kokoro import KPipeline

from consultar_rag import obtener_coleccion, preguntar

# --- Setup de ffmpeg portable (para Whisper) ---
ffmpeg_real_path = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_real_path)
ffmpeg_target = os.path.join(ffmpeg_dir, "ffmpeg.exe")
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

# --- Setup de Kokoro TTS (español, corre en CPU, no usa VRAM) ---
print("Cargando modelo de voz (Kokoro)...")
tts_pipeline = KPipeline(lang_code='e')  # 'e' = español
VOZ_TTS = 'ef_dora'  # voz femenina en español; prueba otras si quieres cambiarla

# --- Configuración de audio ---
fs = 16000
frame_ms = 30
frame_size = int(fs * frame_ms / 1000)
vad = webrtcvad.Vad(2)

HOTKEY = 'ctrl+space'


def limpiar_texto_para_voz(texto):
    """Quita elementos que no deben leerse en voz alta."""
    texto_limpio = re.sub(r'\S+@\S+\.\S+', 'correo electronico', texto)
    texto_limpio = re.sub(r'\(Fuente:.*?\)', '', texto_limpio)
    texto_limpio = re.sub(r'\[Fuente:.*?\]', '', texto_limpio)
    texto_limpio = texto_limpio.replace('*', '').replace('•', '')
    texto_limpio = ' '.join(texto_limpio.split())
    return texto_limpio


def hablar(texto, archivo_salida="respuesta.wav"):
    """Genera audio con Kokoro y lo reproduce."""
    texto_limpio = limpiar_texto_para_voz(texto)
    if not texto_limpio.strip():
        return

    audio_completo = []
    for _, _, audio_chunk in tts_pipeline(texto_limpio, voice=VOZ_TTS):
        audio_completo.append(audio_chunk)

    if not audio_completo:
        print("(No se pudo generar audio)")
        return

    audio_final = np.concatenate(audio_completo)
    sf.write(archivo_salida, audio_final, 24000)  # Kokoro genera a 24kHz

    sd.play(audio_final, 24000)
    sd.wait()


def esperar_hotkey():
    print(f"\nPresiona {HOTKEY} para preguntar (o Ctrl+C para salir)...")
    keyboard.wait(HOTKEY)
    print("¡Escuchando!")


def escuchar_hasta_silencio(silencio_max_ms=1000, max_segundos=30):
    frames_silencio_max = silencio_max_ms // frame_ms
    max_frames = int(max_segundos * 1000 / frame_ms)
    audio_grabado = []
    silencio_contador = 0
    empezo_a_hablar = False
    total_frames = 0

    with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
        while True:
            chunk, _ = stream.read(frame_size)
            chunk_bytes = chunk.tobytes()
            es_voz = vad.is_speech(chunk_bytes, fs)
            total_frames += 1

            if es_voz:
                empezo_a_hablar = True
                silencio_contador = 0
                audio_grabado.append(chunk)
            elif empezo_a_hablar:
                silencio_contador += 1
                audio_grabado.append(chunk)
                if silencio_contador >= frames_silencio_max:
                    break

            if total_frames >= max_frames:
                break

    if not audio_grabado:
        return None

    print("Silencio detectado, transcribiendo...")
    return np.concatenate(audio_grabado)


if __name__ == "__main__":
    print("\nListo. Push-to-talk + RAG + Voz (Kokoro) activo.")
    try:
        while True:
            esperar_hotkey()
            audio = escuchar_hasta_silencio()

            if audio is None:
                print("No se detectó voz.")
                continue

            sf.write("temp.wav", audio, fs)
            resultado = whisper_model.transcribe("temp.wav", language="es")
            pregunta = resultado["text"].strip()

            if not pregunta:
                print("No se entendió nada.")
                continue

            print(f"Pregunta: {pregunta}")
            print("Buscando en documentos...")

            respuesta = preguntar(pregunta, coleccion, mostrar_fuentes=False)
            print(f"\nRespuesta: {respuesta}")

            hablar(respuesta)

    except KeyboardInterrupt:
        print("\nSaliendo...")