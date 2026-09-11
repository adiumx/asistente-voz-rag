"""
Proceso hijo: aquí vive todo el cómputo pesado (Whisper, RAG, Kokoro).

Corre completamente aislado del proceso de la interfaz gráfica. Si PyTorch o
CUDA crashean a nivel nativo (el problema que nos trajo hasta aquí), este
proceso muere solo -- la ventana del asistente sigue viva y puede reportarlo
y, si se desea, reiniciar este proceso.

Protocolo por las colas (todo son tuplas, primer elemento = tipo de mensaje):

  Entrada (main -> este proceso):
    ("CARGAR",)
    ("TURNO", audio_float32_numpy, silenciado_bool)
    ("REINDEXAR",)
    ("SALIR",)

  Salida (este proceso -> main):
    ("CARGADO", n_fragmentos)
    ("ERROR_CARGA", mensaje)
    ("PREGUNTA", texto)
    ("RESPUESTA", texto, fuentes)
    ("AUDIO_TROZO", audio_numpy_float32, samplerate)  # uno por cada trozo de voz sintetizado
    ("AUDIO_FIN",)        # marca que ya no vienen mas AUDIO_TROZO de este turno
    ("SIN_VOZ",)          # hubo respuesta pero no se sintetizo audio
    ("VACIO",)            # no se entendió nada en la transcripción
    ("FIN_TURNO",)        # siempre se manda al terminar un ("TURNO", ...)
    ("ERROR_TURNO", mensaje)
    ("REINDEXADO", ok_bool, mensaje)
"""

import time
import os
import re
import shutil
import platform

import numpy as np

ES_WINDOWS = platform.system() == "Windows"

MODELO_WHISPER = os.environ.get("MORITO_WHISPER", "base")
VOZ_TTS = os.environ.get("MORITO_VOZ", "em_alex")
IDIOMA_TTS = "e"
MAX_CHARS_VOZ = 700

ALUCINACIONES = [
    "subtítulos realizados por la comunidad de amara.org",
    "subtitulos realizados por la comunidad de amara.org",
    "gracias por ver el video", "gracias por ver el vídeo",
    "suscríbete al canal", "suscribete al canal",
    "más información en", "www.", "¡gracias!", "gracias.",
]


def _es_alucinacion(texto):
    t = texto.lower().strip(" .!¡¿?")
    if len(t) < 3:
        return True
    return any(frase in t for frase in ALUCINACIONES)


def _limpiar_para_voz(texto):
    t = re.sub(r"\S+@\S+\.\S+", "correo electrónico", texto)
    t = re.sub(r"\(Fuente:.*?\)", "", t)
    t = re.sub(r"\[Fuente:.*?\]", "", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[*_`#•·|]", " ", t)
    t = re.sub(r"^\s*[-–]\s+", "", t, flags=re.MULTILINE)
    t = t.replace("~", "aproximadamente ")
    t = " ".join(t.split())
    if len(t) > MAX_CHARS_VOZ:
        corte = t.rfind(".", 0, MAX_CHARS_VOZ)
        t = t[:corte + 1] if corte > 100 else t[:MAX_CHARS_VOZ] + "."
    return t


def _preparar_ffmpeg():
    try:
        import imageio_ffmpeg
    except ImportError:
        return
    ruta = imageio_ffmpeg.get_ffmpeg_exe()
    carpeta = os.path.dirname(ruta)
    nombre = "ffmpeg.exe" if ES_WINDOWS else "ffmpeg"
    destino = os.path.join(carpeta, nombre)
    if not os.path.exists(destino):
        try:
            shutil.copy(ruta, destino)
            if not ES_WINDOWS:
                os.chmod(destino, 0o755)
        except OSError:
            pass
    os.environ["PATH"] += os.pathsep + carpeta


def ejecutar(cola_entrada, cola_salida):
    """Punto de entrada del proceso hijo. Bucle de vida completo."""

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    whisper_model = None
    tts = None
    motor = None

    def cargar():
        nonlocal whisper_model, tts, motor
        _preparar_ffmpeg()

        import torch
        import whisper
        from kokoro import KPipeline
        from consultar_rag import MotorRAG

        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[diag] Whisper usara: {dispositivo}", flush=True)
        whisper_model = whisper.load_model(MODELO_WHISPER, device=dispositivo)

        try:
            tts = KPipeline(lang_code=IDIOMA_TTS, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
        except TypeError:
            tts = KPipeline(lang_code=IDIOMA_TTS)

        motor = MotorRAG()
        return motor.contar()

    def reducir_ruido(audio_float32):
        try:
            import noisereduce as nr
            return nr.reduce_noise(y=audio_float32, sr=16000, stationary=True)
        except Exception:
            return audio_float32

    def transcribir(audio_float32):
        audio_limpio = reducir_ruido(audio_float32)
        r = whisper_model.transcribe(
            audio_limpio,
            language="es",
            fp16=False,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        texto = (r.get("text") or "").strip()
        return "" if _es_alucinacion(texto) else texto

    def sintetizar_y_enviar(texto, cola_salida):
        limpio = _limpiar_para_voz(texto)
        if not limpio:
            return False

        algo_enviado = False
        for _, _, audio_chunk in tts(limpio, voice=VOZ_TTS):
            if audio_chunk is not None and len(audio_chunk) > 0:
                cola_salida.put(("AUDIO_TROZO", audio_chunk, 24000))
                algo_enviado = True

        if algo_enviado:
            cola_salida.put(("AUDIO_FIN",))
        return algo_enviado

    # --- ciclo de vida ---
    try:
        cola_salida.put(("CARGADO", cargar()))
    except Exception as e:
        cola_salida.put(("ERROR_CARGA", str(e)))
        return

    while True:
        try:
            msg = cola_entrada.get()
        except (EOFError, OSError):
            break

        tipo = msg[0]

        if tipo == "SALIR":
            break

        elif tipo == "REINDEXAR":
            try:
                import subprocess
                import sys as _sys
                subprocess.run([_sys.executable, "indexar_pdfs.py"], check=True)
                motor = None
                from consultar_rag import MotorRAG
                motor = MotorRAG()
                cola_salida.put(("REINDEXADO", True, f"{motor.contar()} fragmentos"))
            except Exception as e:
                cola_salida.put(("REINDEXADO", False, str(e)))

        elif tipo == "TURNO":
            _, audio_float32, silenciado = msg
            try:
                t0 = time.monotonic()
                pregunta = transcribir(audio_float32)
                print(f"[latencia-detalle] transcripcion: {time.monotonic()-t0:.2f}s", flush=True)
                if not pregunta:
                    cola_salida.put(("VACIO",))
                    cola_salida.put(("FIN_TURNO",))
                    continue

                cola_salida.put(("PREGUNTA", pregunta))

                t1 = time.monotonic()
                texto, fuentes = motor.preguntar(pregunta)
                print(f"[latencia-detalle] rag + gemma: {time.monotonic()-t1:.2f}s", flush=True)
                cola_salida.put(("RESPUESTA", texto, fuentes))

                if not silenciado:
                    t2 = time.monotonic()
                    hubo_audio = sintetizar_y_enviar(texto, cola_salida)
                    print(f"[latencia-detalle] sintesis completa kokoro: {time.monotonic()-t2:.2f}s", flush=True)
                    if not hubo_audio:
                        cola_salida.put(("SIN_VOZ",))
                else:
                    cola_salida.put(("SIN_VOZ",))

            except Exception as e:
                cola_salida.put(("ERROR_TURNO", str(e)))
            finally:
                cola_salida.put(("FIN_TURNO",))
