"""
Morito - asistente de voz con RAG sobre tus documentos.

Interfaz: un orbe flotante siempre encima de las demas ventanas. Clic en el
orbe (o el atajo global) para hablar. Las fuentes nunca se leen en voz alta:
aparecen como chips en la tarjeta lateral.

Arquitectura: el computo pesado (Whisper, RAG, Kokoro) corre en un PROCESO
separado (motor_inferencia.py), no solo un hilo. En Windows, PyTorch puede
crashear a nivel nativo -sin excepcion de Python, sin traceback- al lanzar su
primera inferencia real desde un hilo que no es el principal del proceso.
Aislarlo en su propio proceso evita que ese crash se lleve la interfaz.

Funciona en Windows y en Linux (X11). Ver notas de Wayland en el README.
"""

import os
import sys
import platform
import multiprocessing as mp

import numpy as np
import sounddevice as sd
import webrtcvad

from PySide6.QtCore import QObject, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import QApplication

from orbe import OrbeFlotante, PanelInfo

ES_WINDOWS = platform.system() == "Windows"

ATAJO = os.environ.get("MORITO_ATAJO", "ctrl+space")

FS = 16000
FRAME_MS = 30
FRAME_SIZE = int(FS * FRAME_MS / 1000)

SILENCIO_FIN_MS = 1100
ESPERA_INICIAL_S = 6
MAX_GRABACION_S = 25
ENERGIA_MINIMA = 0.006

TIMEOUT_CARGA_S = 120   # la primera carga de modelos puede tardar
TIMEOUT_TURNO_S = 60    # una respuesta completa no debería tardar mas que esto


# ---------------------------------------------------------------- trabajador

class Trabajador(QObject):
    """
    Vive en su propio QThread. Administra el proceso hijo de inferencia y la
    grabacion de audio (que sabemos que es segura en un hilo: nunca crasheo
    durante todo el diagnostico).
    """

    estado = Signal(str)
    nivel = Signal(float)
    pregunta_lista = Signal(str)
    respuesta_lista = Signal(str, list)
    aviso = Signal(str)
    modelos_listos = Signal(int)

    def __init__(self):
        super().__init__()
        self.vad = webrtcvad.Vad(2)
        self.silenciado = False
        self._ocupado = False
        self._cancelar_reproduccion = False

        self.proceso = None
        self.cola_entrada = None
        self.cola_salida = None

    # ---------- ciclo de vida del proceso hijo ----------

    def _lanzar_proceso(self):
        ctx = mp.get_context("spawn")  # explicito y seguro en Windows y Linux
        self.cola_entrada = ctx.Queue()
        self.cola_salida = ctx.Queue()

        import motor_inferencia
        self.proceso = ctx.Process(
            target=motor_inferencia.ejecutar,
            args=(self.cola_entrada, self.cola_salida),
            daemon=True,
        )
        self.proceso.start()

    @Slot()
    def cargar(self):
        self.estado.emit("cargando")
        try:
            self._lanzar_proceso()
            msg = self.cola_salida.get(timeout=TIMEOUT_CARGA_S)
        except Exception as e:
            self.aviso.emit(f"El motor de IA no respondio al iniciar: {e}")
            self.estado.emit("error")
            return

        if msg[0] == "CARGADO":
            n = msg[1]
            if n == 0:
                self.aviso.emit("No hay documentos indexados. Corre indexar_pdfs.py primero.")
            self.modelos_listos.emit(n)
            self.estado.emit("listo")
        else:
            self.aviso.emit(f"Error al cargar modelos: {msg[1] if len(msg) > 1 else msg}")
            self.estado.emit("error")

    def _proceso_vivo(self):
        return self.proceso is not None and self.proceso.is_alive()

    def _reiniciar_proceso_si_murio(self):
        if not self._proceso_vivo():
            self.aviso.emit("El motor de IA se detuvo inesperadamente. Reiniciando...")
            self.estado.emit("cargando")
            self.cargar()
            return True
        return False

    # ---------- turno completo ----------

    @Slot()
    def procesar_turno(self):
        if self._ocupado:
            return
        self._ocupado = True
        self._cancelar_reproduccion = False

        try:
            if self._reiniciar_proceso_si_murio():
                return

            audio = self._grabar()
            if audio is None:
                self.estado.emit("listo")
                return

            self.estado.emit("pensando")
            audio_float = audio.astype(np.float32) / 32768.0
            self.cola_entrada.put(("TURNO", audio_float, self.silenciado))

            self._recibir_turno()
            self.estado.emit("listo")

        except Exception as e:
            self.aviso.emit(f"Error: {e}")
            self.estado.emit("error")
            QTimer.singleShot(2500, lambda: self.estado.emit("listo"))
        finally:
            self._ocupado = False
            self.nivel.emit(0.0)

    def _recibir_turno(self):
        """Lee mensajes del proceso hijo hasta ver FIN_TURNO."""
        while True:
            try:
                msg = self.cola_salida.get(timeout=TIMEOUT_TURNO_S)
            except Exception:
                self.aviso.emit("El motor de IA tardo demasiado en responder.")
                return

            tipo = msg[0]

            if tipo == "PREGUNTA":
                self.pregunta_lista.emit(msg[1])

            elif tipo == "RESPUESTA":
                self.respuesta_lista.emit(msg[1], msg[2])

            elif tipo == "AUDIO":
                if not self._cancelar_reproduccion:
                    self.estado.emit("hablando")
                    self._reproducir(msg[1], msg[2])

            elif tipo == "SIN_VOZ":
                pass

            elif tipo == "VACIO":
                self.aviso.emit("No te entendi bien.")

            elif tipo == "ERROR_TURNO":
                self.aviso.emit(f"Error: {msg[1]}")

            elif tipo == "FIN_TURNO":
                return

    def _reproducir(self, audio, samplerate):
        try:
            sd.play(audio, samplerate)
            while sd.get_stream().active:
                if self._cancelar_reproduccion:
                    sd.stop()
                    break
                sd.sleep(50)
        except Exception:
            pass

    # ---------- grabacion (probado: seguro en este hilo) ----------

    def _grabar(self):
        self.estado.emit("grabando")
        frames_silencio_max = SILENCIO_FIN_MS // FRAME_MS
        max_frames = int(MAX_GRABACION_S * 1000 / FRAME_MS)
        frames_espera = int(ESPERA_INICIAL_S * 1000 / FRAME_MS)

        grabado = []
        silencio = 0
        hablo = False
        total = 0

        try:
            stream = sd.InputStream(samplerate=FS, channels=1, dtype="int16",
                                    blocksize=FRAME_SIZE)
        except Exception as e:
            self.aviso.emit(f"No pude abrir el microfono: {e}")
            return None

        with stream:
            while True:
                try:
                    chunk, _ = stream.read(FRAME_SIZE)
                except Exception:
                    break

                total += 1
                muestras = chunk.flatten()

                rms = float(np.sqrt(np.mean((muestras.astype(np.float32) / 32768.0) ** 2)))
                self.nivel.emit(min(1.0, rms * 12))

                try:
                    es_voz = self.vad.is_speech(chunk.tobytes(), FS)
                except Exception:
                    es_voz = rms > ENERGIA_MINIMA

                if es_voz:
                    hablo = True
                    silencio = 0
                    grabado.append(muestras)
                elif hablo:
                    silencio += 1
                    grabado.append(muestras)
                    if silencio >= frames_silencio_max:
                        break

                if not hablo and total >= frames_espera:
                    return None
                if total >= max_frames:
                    break

        if not grabado:
            return None

        audio = np.concatenate(grabado)
        if len(audio) < FS * 0.4:
            return None
        if float(np.sqrt(np.mean((audio.astype(np.float32) / 32768.0) ** 2))) < ENERGIA_MINIMA:
            return None
        return audio

    # ---------- control ----------

    @Slot(bool)
    def set_silencio(self, valor):
        self.silenciado = valor

    def interrumpir(self):
        self._cancelar_reproduccion = True
        try:
            sd.stop()
        except Exception:
            pass

    def reindexar(self):
        if self._proceso_vivo():
            self.cola_entrada.put(("REINDEXAR",))

    def apagar(self):
        self.interrumpir()
        if self._proceso_vivo():
            try:
                self.cola_entrada.put(("SALIR",))
                self.proceso.join(timeout=3)
            except Exception:
                pass
            if self.proceso.is_alive():
                self.proceso.terminate()


# ---------------------------------------------------------------- aplicacion

class Morito(QObject):

    pedir_turno = Signal()
    pedir_carga = Signal()

    def __init__(self, app):
        super().__init__()
        self.app = app

        self.orbe = OrbeFlotante()
        self.panel = PanelInfo()
        self.ultima_pregunta = ""
        self.ultima_respuesta = ""
        self.ultimas_fuentes = []
        self._estado = "cargando"

        self.hilo = QThread()
        self.trabajador = Trabajador()
        self.trabajador.moveToThread(self.hilo)

        self.trabajador.estado.connect(self._cambio_estado)
        self.trabajador.nivel.connect(self.orbe.set_nivel)
        self.trabajador.pregunta_lista.connect(self._nueva_pregunta)
        self.trabajador.respuesta_lista.connect(self._nueva_respuesta)
        self.trabajador.aviso.connect(self._mostrar_aviso)
        self.trabajador.modelos_listos.connect(self._modelos_listos)

        self.pedir_carga.connect(self.trabajador.cargar)
        self.pedir_turno.connect(self.trabajador.procesar_turno)
        self.orbe.silencio_alternado.connect(self.trabajador.set_silencio)

        self.orbe.clicado.connect(self._activar)
        self.orbe.salir_solicitado.connect(self.salir)
        self.orbe.reindexar_solicitado.connect(self._reindexar)

        self.orbe.enterEvent = self._wrap_enter(self.orbe.enterEvent)
        self.orbe.leaveEvent = self._wrap_leave(self.orbe.leaveEvent)

        self.hilo.start()
        self.orbe.show()
        self.pedir_carga.emit()

        self._instalar_atajo()

    def _wrap_enter(self, original):
        def handler(e):
            original(e)
            if self.ultima_respuesta:
                self.panel.cancelar_autocierre()
                self.panel.aparecer(self.orbe)
        return handler

    def _wrap_leave(self, original):
        def handler(e):
            original(e)
            self.panel.desvanecer()
        return handler

    def _cambio_estado(self, estado):
        self._estado = estado
        self.orbe.set_estado(estado)

    def _modelos_listos(self, n):
        self.orbe.setToolTip(f"Morito - {n} fragmentos indexados")

    def _nueva_pregunta(self, texto):
        self.ultima_pregunta = texto
        self.panel.actualizar(texto, "Pensando...", [])

    def _nueva_respuesta(self, texto, fuentes):
        self.ultima_respuesta = texto
        self.ultimas_fuentes = fuentes
        self.orbe.set_n_fuentes(len(fuentes))
        self.panel.actualizar(self.ultima_pregunta, texto, fuentes)
        self.panel.aparecer(self.orbe, segundos=7)

    def _mostrar_aviso(self, texto):
        self.panel.actualizar(self.ultima_pregunta, texto, [])
        self.panel.aparecer(self.orbe, segundos=4)

    def _activar(self):
        if self._estado == "hablando":
            self.trabajador.interrumpir()
            self._cambio_estado("listo")
            return
        if self._estado in ("listo", "error"):
            self.pedir_turno.emit()

    def _reindexar(self):
        self._mostrar_aviso("Reindexando documentos...")
        self.trabajador.reindexar()

    def _instalar_atajo(self):
        try:
            import keyboard
            keyboard.add_hotkey(ATAJO, lambda: QTimer.singleShot(0, self._activar))
        except Exception:
            print(f"[info] Atajo global no disponible. Usa clic en el orbe. "
                  f"(en Linux puede requerir sudo, o define MORITO_ATAJO)")

    def salir(self):
        self.trabajador.apagar()
        self.hilo.quit()
        self.hilo.wait(3000)
        self.app.quit()


def main():
    def excepcion_no_capturada(tipo, valor, tb):
        import traceback
        print("=== ERROR NO CAPTURADO ===")
        traceback.print_exception(tipo, valor, tb)
    sys.excepthook = excepcion_no_capturada

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    _morito = Morito(app)
    sys.exit(app.exec())


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()