"""
Interfaz gráfica flotante del asistente.

- OrbeFlotante: círculo animado, sin marco, siempre encima de todas las ventanas.
- PanelInfo: tarjeta lateral que aparece al pasar el mouse (o unos segundos tras
  responder) mostrando la respuesta y las fuentes. Se oculta sola.

Ambos son ventanas independientes: así el área transparente alrededor del orbe
no roba clics a la aplicación que esté debajo.
"""

import json
import math
import os

from PySide6.QtCore import (Qt, QTimer, QPoint, Signal, QPropertyAnimation,
                            QEasingCurve, QRectF)
from PySide6.QtGui import (QPainter, QColor, QRadialGradient, QPen, QBrush,
                           QFont, QPainterPath, QAction)
from PySide6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                               QMenu, QFrame, QSizePolicy)

ARCHIVO_POSICION = ".orbe_posicion.json"

# Paleta por estado: (color principal, color de brillo)
COLORES = {
    "cargando":  (QColor(120, 130, 150), QColor(120, 130, 150, 60)),
    "listo":     (QColor(96, 112, 140),  QColor(120, 150, 200, 55)),
    "grabando":  (QColor(34, 211, 238),  QColor(34, 211, 238, 90)),
    "pensando":  (QColor(167, 139, 250), QColor(167, 139, 250, 90)),
    "hablando":  (QColor(52, 211, 153),  QColor(52, 211, 153, 90)),
    "error":     (QColor(248, 113, 113), QColor(248, 113, 113, 90)),
}

LEYENDAS = {
    "cargando": "Cargando modelos…",
    "listo": "Listo — clic para hablar",
    "grabando": "Escuchando…",
    "pensando": "Pensando…",
    "hablando": "Respondiendo…",
    "error": "Algo falló",
}


class OrbeFlotante(QWidget):
    """Círculo animado siempre encima. Clic izquierdo = activar/interrumpir."""

    clicado = Signal()
    salir_solicitado = Signal()
    reindexar_solicitado = Signal()
    silencio_alternado = Signal(bool)

    TAM = 96          # lado de la ventana
    RADIO_BASE = 26   # radio del círculo interior

    def __init__(self):
        super().__init__()

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool                      # fuera de la barra de tareas
            | Qt.WindowDoesNotAcceptFocus  # no roba el foco a otras apps
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(self.TAM, self.TAM)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu_contextual)

        self._estado = "cargando"
        self._fase = 0.0
        self._nivel = 0.0        # nivel de micrófono suavizado (0..1)
        self._nivel_objetivo = 0.0
        self._n_fuentes = 0
        self._hover = False
        self._silenciado = False

        self._arrastre = None
        self._se_movio = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)  # ~60 fps

        self._restaurar_posicion()

    # ---------- estado ----------

    def set_estado(self, estado):
        if estado in COLORES:
            self._estado = estado
            self.setToolTip(LEYENDAS.get(estado, ""))
            self.update()

    def set_nivel(self, nivel):
        self._nivel_objetivo = max(0.0, min(1.0, nivel))

    def set_n_fuentes(self, n):
        self._n_fuentes = n
        self.update()

    def esta_silenciado(self):
        return self._silenciado

    # ---------- animación ----------

    def _tick(self):
        self._fase += 0.045
        # suavizado exponencial del nivel de audio, para que no parpadee
        self._nivel += (self._nivel_objetivo - self._nivel) * 0.25
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        centro = QPoint(self.TAM // 2, self.TAM // 2)
        cx, cy = centro.x(), centro.y()
        color, brillo = COLORES[self._estado]

        respiracion = (math.sin(self._fase) + 1) / 2  # 0..1

        # --- halo exterior según estado ---
        if self._estado == "grabando":
            radio_halo = self.RADIO_BASE + 8 + self._nivel * 16
            self._dibujar_halo(p, cx, cy, radio_halo, brillo)
        elif self._estado == "hablando":
            for i in range(3):
                avance = (self._fase * 0.5 + i / 3) % 1.0
                radio_onda = self.RADIO_BASE + avance * 20
                alfa = int(90 * (1 - avance))
                c = QColor(brillo)
                c.setAlpha(alfa)
                p.setPen(QPen(c, 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(centro, int(radio_onda), int(radio_onda))
        elif self._estado in ("listo", "cargando"):
            self._dibujar_halo(p, cx, cy, self.RADIO_BASE + 4 + respiracion * 5, brillo)
        elif self._estado == "error":
            self._dibujar_halo(p, cx, cy, self.RADIO_BASE + 6 + respiracion * 4, brillo)

        # --- arco giratorio mientras piensa ---
        if self._estado in ("pensando", "cargando"):
            radio_arco = self.RADIO_BASE + 9
            rect = QRectF(cx - radio_arco, cy - radio_arco, radio_arco * 2, radio_arco * 2)
            pluma = QPen(color, 3)
            pluma.setCapStyle(Qt.RoundCap)
            p.setPen(pluma)
            p.setBrush(Qt.NoBrush)
            inicio = int((-self._fase * 180 / math.pi * 1.4) % 360) * 16
            p.drawArc(rect, inicio, 100 * 16)

        # --- cuerpo del orbe ---
        escala = 1.0
        if self._estado == "listo":
            escala = 1.0 + respiracion * 0.05
        elif self._estado == "grabando":
            escala = 1.0 + self._nivel * 0.18
        elif self._estado == "hablando":
            escala = 1.0 + abs(math.sin(self._fase * 2)) * 0.08
        if self._hover:
            escala *= 1.08

        radio = self.RADIO_BASE * escala

        grad = QRadialGradient(cx - radio * 0.3, cy - radio * 0.35, radio * 1.7)
        grad.setColorAt(0.0, color.lighter(150))
        grad.setColorAt(0.6, color)
        grad.setColorAt(1.0, color.darker(160))

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(centro, int(radio), int(radio))

        # brillo especular
        p.setBrush(QColor(255, 255, 255, 55))
        p.drawEllipse(QPoint(int(cx - radio * 0.32), int(cy - radio * 0.36)),
                      int(radio * 0.28), int(radio * 0.2))

        # --- icono de silencio ---
        if self._silenciado:
            p.setPen(QPen(QColor(255, 255, 255, 190), 2.5, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(int(cx - radio * 0.45), int(cy - radio * 0.45),
                       int(cx + radio * 0.45), int(cy + radio * 0.45))

        # --- insignia con el número de fuentes ---
        if self._n_fuentes > 0 and self._estado in ("listo", "hablando"):
            self._dibujar_insignia(p, cx, cy, radio)

    def _dibujar_halo(self, p, cx, cy, radio, color_brillo):
        grad = QRadialGradient(cx, cy, radio * 1.6)
        grad.setColorAt(0.45, QColor(color_brillo.red(), color_brillo.green(),
                                     color_brillo.blue(), color_brillo.alpha()))
        grad.setColorAt(1.0, QColor(color_brillo.red(), color_brillo.green(),
                                    color_brillo.blue(), 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(QPoint(cx, cy), int(radio * 1.6), int(radio * 1.6))

    def _dibujar_insignia(self, p, cx, cy, radio):
        r = 9
        bx = int(cx + radio * 0.72)
        by = int(cy - radio * 0.72)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(30, 34, 44, 230))
        p.drawEllipse(QPoint(bx, by), r, r)
        p.setPen(QPen(QColor(255, 255, 255, 210), 1))
        f = QFont()
        f.setPointSize(7)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(bx - r, by - r, r * 2, r * 2),
                   Qt.AlignCenter, str(self._n_fuentes))

    # ---------- interacción ----------

    def enterEvent(self, _):
        self._hover = True
        self.update()

    def leaveEvent(self, _):
        self._hover = False
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._arrastre = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._se_movio = False

    def mouseMoveEvent(self, e):
        if self._arrastre is not None and (e.buttons() & Qt.LeftButton):
            self.move(e.globalPosition().toPoint() - self._arrastre)
            self._se_movio = True

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            if not self._se_movio:
                self.clicado.emit()
            else:
                self._guardar_posicion()
            self._arrastre = None

    def _menu_contextual(self, punto):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background:#20242e; color:#e6e9f0; border:1px solid #333a48;
                    border-radius:8px; padding:6px; }
            QMenu::item { padding:6px 18px; border-radius:5px; }
            QMenu::item:selected { background:#2f3646; }
        """)

        a_silencio = QAction("Reactivar voz" if self._silenciado else "Silenciar voz", self)
        a_silencio.triggered.connect(self._alternar_silencio)
        menu.addAction(a_silencio)

        a_reindex = QAction("Reindexar documentos", self)
        a_reindex.triggered.connect(self.reindexar_solicitado.emit)
        menu.addAction(a_reindex)

        menu.addSeparator()
        a_salir = QAction("Salir", self)
        a_salir.triggered.connect(self.salir_solicitado.emit)
        menu.addAction(a_salir)

        menu.exec(self.mapToGlobal(punto))

    def _alternar_silencio(self):
        self._silenciado = not self._silenciado
        self.silencio_alternado.emit(self._silenciado)
        self.update()

    # ---------- posición persistente ----------

    def _guardar_posicion(self):
        try:
            with open(ARCHIVO_POSICION, "w") as f:
                json.dump({"x": self.x(), "y": self.y()}, f)
        except OSError:
            pass

    def _restaurar_posicion(self):
        pantalla = self.screen().availableGeometry()
        x = pantalla.right() - self.TAM - 32
        y = pantalla.bottom() - self.TAM - 60

        if os.path.exists(ARCHIVO_POSICION):
            try:
                with open(ARCHIVO_POSICION) as f:
                    pos = json.load(f)
                # solo si sigue dentro de la pantalla actual
                if pantalla.contains(QPoint(pos["x"], pos["y"])):
                    x, y = pos["x"], pos["y"]
            except (OSError, ValueError, KeyError):
                pass

        self.move(x, y)


class PanelInfo(QWidget):
    """
    Tarjeta discreta junto al orbe: última pregunta, respuesta y fuentes.
    Aparece al pasar el mouse sobre el orbe o unos segundos tras responder.
    """

    ANCHO = 340

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedWidth(self.ANCHO)

        tarjeta = QFrame(self)
        tarjeta.setObjectName("tarjeta")
        tarjeta.setStyleSheet("""
            QFrame#tarjeta {
                background: rgba(24, 27, 35, 238);
                border: 1px solid rgba(255,255,255,26);
                border-radius: 14px;
            }
            QLabel { color: #e6e9f0; }
            QLabel#pregunta { color: #8b93a7; font-size: 11px; }
            QLabel#respuesta { font-size: 13px; }
            QLabel#chip {
                background: rgba(255,255,255,16);
                border-radius: 8px;
                padding: 3px 9px;
                color: #a9b2c7;
                font-size: 10px;
            }
        """)

        self.lbl_pregunta = QLabel("", objectName="pregunta")
        self.lbl_pregunta.setWordWrap(True)

        self.lbl_respuesta = QLabel("", objectName="respuesta")
        self.lbl_respuesta.setWordWrap(True)
        self.lbl_respuesta.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.fila_fuentes = QHBoxLayout()
        self.fila_fuentes.setSpacing(6)
        self.fila_fuentes.setContentsMargins(0, 0, 0, 0)
        self.fila_fuentes.addStretch()

        interior = QVBoxLayout(tarjeta)
        interior.setContentsMargins(16, 14, 16, 14)
        interior.setSpacing(8)
        interior.addWidget(self.lbl_pregunta)
        interior.addWidget(self.lbl_respuesta)
        interior.addLayout(self.fila_fuentes)

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.addWidget(tarjeta)

        self._anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.InOutQuad)

        self._timer_ocultar = QTimer(self)
        self._timer_ocultar.setSingleShot(True)
        self._timer_ocultar.timeout.connect(self.desvanecer)

        self.setWindowOpacity(0.0)

    def actualizar(self, pregunta, respuesta, fuentes):
        self.lbl_pregunta.setText(f"“{pregunta}”" if pregunta else "")

        texto = respuesta or ""
        if len(texto) > 420:
            texto = texto[:417].rstrip() + "…"
        self.lbl_respuesta.setText(texto)

        # limpia los chips anteriores
        while self.fila_fuentes.count() > 1:
            item = self.fila_fuentes.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for f in (fuentes or [])[:3]:
            nombre = os.path.splitext(f["archivo"])[0]
            if len(nombre) > 22:
                nombre = nombre[:21] + "…"
            chip = QLabel(f"{nombre} · p.{f['pagina']}", objectName="chip")
            self.fila_fuentes.insertWidget(self.fila_fuentes.count() - 1, chip)

        self.adjustSize()

    def colocar_junto_a(self, orbe):
        pantalla = orbe.screen().availableGeometry()
        self.adjustSize()

        # por defecto a la izquierda del orbe; si no cabe, a la derecha
        x = orbe.x() - self.width() - 12
        if x < pantalla.left() + 8:
            x = orbe.x() + orbe.width() + 12

        y = orbe.y() + orbe.height() // 2 - self.height() // 2
        y = max(pantalla.top() + 8, min(y, pantalla.bottom() - self.height() - 8))
        self.move(x, y)

    def aparecer(self, orbe, segundos=0):
        if not self.lbl_respuesta.text():
            return
        self.colocar_junto_a(orbe)
        self.show()
        self._anim.stop()
        self._anim.setStartValue(self.windowOpacity())
        self._anim.setEndValue(1.0)
        self._anim.start()

        self._timer_ocultar.stop()
        if segundos > 0:
            self._timer_ocultar.start(int(segundos * 1000))

    def desvanecer(self):
        self._anim.stop()
        self._anim.setStartValue(self.windowOpacity())
        self._anim.setEndValue(0.0)
        self._anim.start()
        QTimer.singleShot(200, self.hide)

    def cancelar_autocierre(self):
        self._timer_ocultar.stop()
