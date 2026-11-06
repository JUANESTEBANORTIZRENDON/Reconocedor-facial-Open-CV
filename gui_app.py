"""Interfaz gráfica del sistema de reconocimiento facial (tema Matrix).

Reúne en una sola ventana el flujo completo:
  01 REGISTRAR ROSTRO       -> pide un nombre y captura muestras desde la cámara
  02 ENTRENAR MODELO        -> entrena el reconocedor LBPH con las muestras de data/
  03 VALIDAR RECONOCIMIENTO -> reconoce rostros en tiempo real

Reutiliza la lógica de capture_faces.py, train_model.py y recognize_faces.py.
Solo usa tkinter (incluido con Python), OpenCV y numpy.
Ejecutar:  python gui_app.py     (Esc detiene la operación en curso)
"""
import contextlib
import ctypes
import json
import queue
import random
import sys
import threading
import time
import tkinter as tk

import cv2
import numpy as np

import capture_faces
import recognize_faces
import train_model

# ----------------------------------------------------------------- paleta
BG = "#020a03"
PANEL = "#04140a"
PANEL_HI = "#082a12"
GREEN = "#00ff41"
GREEN_MID = "#00c030"
GREEN_DIM = "#008f11"
GREEN_DARK = "#003b00"
TEXT_DIM = "#4fa862"
WHITEGREEN = "#c9ffd6"
RED = "#ff3b3b"
AMBER = "#ffd34d"

IDLE, CAPTURE, RECOGNIZE, TRAIN = "idle", "capture", "recognize", "train"
S = 1.0     # factor de escala según los DPI de la pantalla (se calcula en App)


def px(n):
    """Convierte píxeles de diseño (96 DPI) a píxeles reales de la pantalla."""
    return int(round(n * S))


def F(size, bold=False):
    """Fuente Consolas con tamaño en píxeles (el negativo indica píxeles en Tk)."""
    return ("Consolas", -px(size), "bold") if bold else ("Consolas", -px(size))


def enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# ------------------------------------------------------- dibujo sobre cámara
def draw_face_box(img, x, y, w, h, color, label=None, sublabel=None):
    """Caja con esquinas tipo 'visor' y etiqueta opcional."""
    dim = tuple(int(c * 0.35) for c in color)
    cv2.rectangle(img, (x, y), (x + w, y + h), dim, 1)
    arm = max(14, min(w, h) // 4)
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * arm, cy), color, 3)
        cv2.line(img, (cx, cy), (cx, cy + dy * arm), color, 3)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top = max(0, y - th - 14)
        cv2.rectangle(img, (x, top), (x + tw + 12, top + th + 10), (0, 0, 0), -1)
        cv2.rectangle(img, (x, top), (x + tw + 12, top + th + 10), color, 1)
        cv2.putText(img, label, (x + 6, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if sublabel:
        cv2.putText(img, sublabel, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def fit_to_canvas(frame, width, height):
    """Ajusta el frame al tamaño del lienzo conservando la proporción (bandas negras)."""
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (height - nh) // 2, (width - nw) // 2
    return cv2.copyMakeBorder(resized, top, height - nh - top, left, width - nw - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


# ---------------------------------------------------------------- lluvia
class MatrixRain:
    """Lluvia de caracteres. Un búfer que se va oscureciendo deja la 'estela'."""

    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+=*#<>/\\|"

    def __init__(self, width, height, cell, scale):
        self.cell, self.scale = cell, scale
        self.cols, self.rows = width // cell, height // cell
        self.buf = np.zeros((height, width, 3), np.uint8)
        self.pos = np.array([random.uniform(-self.rows, 0) for _ in range(self.cols)])
        self.speed = np.random.uniform(0.25, 0.8, self.cols)

    def step(self):
        # Azul y rojo se apagan más rápido que el verde: blanco -> verde -> negro
        self.buf = cv2.subtract(self.buf, (14, 7, 14, 0))
        for i in range(self.cols):
            before = int(self.pos[i])
            self.pos[i] += self.speed[i]
            row = int(self.pos[i])
            if row != before and 0 <= row < self.rows:
                ch = random.choice(self.CHARS)
                cv2.putText(self.buf, ch, (i * self.cell, row * self.cell + self.cell - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * self.scale, (190, 255, 190), 1)
            if self.pos[i] > self.rows + 4 and random.random() < 0.05:
                self.pos[i] = random.uniform(-self.rows * 0.6, 0)
                self.speed[i] = random.uniform(0.25, 0.8)
        return self.buf


# --------------------------------------------------------------- botón neón
class NeonButton(tk.Canvas):
    """Botón con esquinas cortadas, brillo al pasar el mouse y estado de atención."""

    PALETTES = {
        "primary": dict(fill=PANEL, outline=GREEN_DIM, label=GREEN, tag=GREEN_DIM,
                        h_fill=PANEL_HI, h_outline=GREEN, h_label=WHITEGREEN,
                        p_fill=GREEN, p_label=BG),
        "danger": dict(fill=PANEL, outline="#7a1a1a", label="#ff6b6b", tag="#7a1a1a",
                       h_fill="#240808", h_outline=RED, h_label="#ffd0d0",
                       p_fill=RED, p_label=BG),
    }

    def __init__(self, master, tag, text, command, width, height, variant="primary", bg=BG):
        super().__init__(master, width=width, height=height, bg=bg,
                         highlightthickness=0, cursor="hand2")
        self.w, self.h = width, height
        self.tag_text, self.text, self.command = tag, text, command
        self.pal = self.PALETTES[variant]
        self.enabled, self.hover, self.pressed = True, False, False
        self.attention, self._blink_on, self._blinking = False, False, False
        self.bind("<Enter>", lambda e: self._set(hover=True))
        self.bind("<Leave>", lambda e: self._set(hover=False, pressed=False))
        self.bind("<ButtonPress-1>", lambda e: self._set(pressed=True))
        self.bind("<ButtonRelease-1>", self._release)
        self._draw()

    def _set(self, **kw):
        if "hover" in kw:
            self.hover = kw["hover"]
        if "pressed" in kw:
            self.pressed = kw["pressed"]
        self._draw()

    def _release(self, event):
        was_pressed = self.pressed
        self._set(pressed=False)
        inside = 0 <= event.x <= self.w and 0 <= event.y <= self.h
        if self.enabled and was_pressed and inside and self.command:
            self.command()

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        if not enabled:
            self.hover = self.pressed = False
        self._draw()

    def set_attention(self, on):
        """Parpadea el borde para indicar al usuario el siguiente paso recomendado."""
        self.attention = on
        if on and not self._blinking:
            self._blinking = True
            self._pulse()
        self._draw()

    def _pulse(self):
        if not self.attention:
            self._blinking = False
            self._blink_on = False
            self._draw()
            return
        self._blink_on = not self._blink_on
        self._draw()
        self.after(450, self._pulse)

    def _draw(self):
        self.delete("all")
        p = self.pal
        if not self.enabled:
            fill, outline, label, tag, width = BG, "#0d2a12", "#1d4d29", "#123a1c", 1
        elif self.pressed:
            fill, outline, label, tag, width = p["p_fill"], p["h_outline"], p["p_label"], p["p_label"], 2
        elif self.hover:
            fill, outline, label, tag, width = p["h_fill"], p["h_outline"], p["h_label"], p["h_outline"], 2
        elif self.attention and self._blink_on:
            fill, outline, label, tag, width = p["h_fill"], p["h_outline"], p["h_label"], p["h_outline"], 2
        else:
            fill, outline, label, tag, width = p["fill"], p["outline"], p["label"], p["tag"], 1
        pad, cut = 2, px(10)
        w, h = self.w, self.h
        pts = [pad + cut, pad, w - pad, pad, w - pad, h - pad - cut,
               w - pad - cut, h - pad, pad, h - pad, pad, pad + cut]
        self.create_polygon(pts, fill=fill, outline=outline, width=width)
        self.create_text(px(16), h // 2, text=self.tag_text, anchor="w", fill=tag, font=F(14, True))
        self.create_text(px(54), h // 2, text=self.text, anchor="w", fill=label, font=F(14, True))
        if self.attention and self.enabled:
            self.create_text(w - px(14), h // 2, text="◄", anchor="e", fill=outline, font=F(14))


# ------------------------------------------------------- diálogo de nombre
class NameDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.title("Registrar rostro")
        self.transient(parent)
        self.resizable(False, False)
        self.result = None

        border = tk.Frame(self, bg=GREEN_DIM, padx=1, pady=1)
        border.pack(padx=px(14), pady=px(14))
        box = tk.Frame(border, bg=PANEL, padx=px(24), pady=px(20))
        box.pack()

        tk.Label(box, text="// NUEVO REGISTRO", font=F(18, True), fg=GREEN, bg=PANEL).pack(anchor="w")
        tk.Label(box, text="Ingrese el nombre de la persona a registrar:", font=F(13),
                 fg=TEXT_DIM, bg=PANEL).pack(anchor="w", pady=(px(8), px(10)))

        self.var = tk.StringVar()
        self.entry = tk.Entry(box, textvariable=self.var, width=30, font=F(18), bg="#001a05", fg=GREEN,
                              insertbackground=GREEN, selectbackground=GREEN_DARK, selectforeground=GREEN,
                              relief="flat", highlightthickness=1, highlightbackground=GREEN_DIM,
                              highlightcolor=GREEN)
        self.entry.pack(fill="x", ipady=px(6))

        self.hint = tk.Label(box, text="", font=F(12), fg=TEXT_DIM, bg=PANEL, anchor="w")
        self.hint.pack(fill="x", pady=(px(8), px(14)))
        self.var.trace_add("write", self._on_change)
        self._on_change()

        row = tk.Frame(box, bg=PANEL)
        row.pack(fill="x")
        NeonButton(row, ">>", "INICIAR", self._ok, px(170), px(44), bg=PANEL).pack(side="left")
        NeonButton(row, "XX", "CANCELAR", self.destroy, px(190), px(44), variant="danger",
                   bg=PANEL).pack(side="right")

        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 3
        self.geometry(f"+{max(0, x)}+{max(0, y)}")
        self.grab_set()
        self.entry.focus_set()

    def _on_change(self, *_):
        clean = capture_faces.clean_name(self.var.get().strip())
        if clean:
            self.hint.config(text=f"Se guardará en: data/{clean}/", fg=TEXT_DIM)
        else:
            self.hint.config(text="Use letras o números (sin símbolos)", fg=TEXT_DIM)

    def _ok(self):
        if not capture_faces.clean_name(self.var.get().strip()):
            self.hint.config(text="El nombre no puede estar vacío", fg=RED)
            return
        self.result = self.var.get().strip()
        self.destroy()


class _QueueWriter:
    """Convierte los print() del entrenamiento (en otro hilo) en mensajes para la consola."""

    def __init__(self, q):
        self.q, self.buf = q, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.q.put(("log", line.rstrip()))
        return len(s)

    def flush(self):
        pass


# ------------------------------------------------------------ aplicación
class App(tk.Tk):
    def __init__(self):
        enable_dpi_awareness()
        super().__init__()
        global S
        S = max(1.0, min(2.0, self.winfo_fpixels("1i") / 96.0))

        self.title("FACE//MATRIX · Reconocimiento facial")
        self.configure(bg=BG)
        self.mode = IDLE
        self.cam = None
        self.q = queue.Queue()
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.recognizer, self.labels = None, {}
        self.cap = dict(name="", dir=None, count=0, saved=0, frame_id=0)
        self.progress_value, self.progress_text = 0.0, "—"
        self._photo = None
        self._last_clock = ""

        self.cw, self.ch = px(576), px(432)
        self.rain = MatrixRain(self.cw, self.ch, px(14), S)
        self._build_ui()
        self._center()

        self.bind("<Escape>", lambda e: self.on_stop())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.log("Sistema iniciado.", "ok")
        if self.detector.empty():
            self.log("Error: no se pudo cargar el Haar Cascade.", "err")
        if not hasattr(cv2, "face"):
            self.log("Error: falta cv2.face. Instale opencv-contrib-python (y desinstale opencv-python).", "err")
        self.refresh_status(initial=True)
        self._set_mode(IDLE)
        self.after(50, self._loop)
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------ interfaz
    def _build_ui(self):
        pad = px(16)

        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=pad, pady=(pad, 0))
        tk.Label(header, text="FACE//MATRIX", font=F(28, True), fg=GREEN, bg=BG).pack(side="left")
        tk.Label(header, text="   RECONOCIMIENTO FACIAL · OPENCV + LBPH", font=F(12),
                 fg=GREEN_DIM, bg=BG).pack(side="left", pady=(px(12), 0))
        self.clock = tk.Label(header, text="", font=F(16), fg=GREEN_MID, bg=BG)
        self.clock.pack(side="right", pady=(px(8), 0))
        tk.Frame(self, bg=GREEN_DARK, height=px(2)).pack(fill="x", padx=pad, pady=(px(8), px(14)))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=pad, pady=(0, pad))
        side = tk.Frame(body, bg=BG)
        side.pack(side="left", fill="y", padx=(0, pad))
        main = tk.Frame(body, bg=BG)
        main.pack(side="left", fill="both", expand=True)

        # --- barra lateral
        bw, bh = px(310), px(50)
        self._section(side, "// OPERACIONES")
        self.btn_register = NeonButton(side, "01", "REGISTRAR ROSTRO", self.on_register, bw, bh)
        self.btn_train = NeonButton(side, "02", "ENTRENAR MODELO", self.on_train, bw, bh)
        self.btn_recog = NeonButton(side, "03", "VALIDAR RECONOCIMIENTO", self.on_recognize, bw, bh)
        self.btn_stop = NeonButton(side, "XX", "DETENER", self.on_stop, bw, px(42), variant="danger")
        for b, gap in ((self.btn_register, 0), (self.btn_train, 8), (self.btn_recog, 8), (self.btn_stop, 14)):
            b.pack(pady=(px(gap), 0))

        self._section(side, "// ESTADO", top=18)
        grid = tk.Frame(side, bg=BG)
        grid.pack(fill="x")
        self.lbl_cam = self._status_row(grid, 0, "CÁMARA")
        self.lbl_model = self._status_row(grid, 1, "MODELO")
        self.lbl_people = self._status_row(grid, 2, "PERSONAS")

        self._section(side, "// REGISTRADOS", top=18)
        self.people_box = tk.Text(side, height=6, width=1, bg=PANEL, fg=GREEN, font=F(13), relief="flat",
                                  highlightthickness=1, highlightbackground=GREEN_DARK, state="disabled",
                                  cursor="arrow", padx=px(8), pady=px(6), wrap="none")
        self.people_box.pack(fill="x")
        self.people_box.tag_configure("name", foreground=GREEN)
        self.people_box.tag_configure("count", foreground=GREEN_DIM)
        self.people_box.tag_configure("empty", foreground=TEXT_DIM)

        # --- zona principal: video, progreso, consola
        frame = tk.Frame(main, bg=GREEN_DIM, padx=1, pady=1)
        frame.pack()
        self.canvas = tk.Canvas(frame, width=self.cw, height=self.ch, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.img_item = self.canvas.create_image(0, 0, anchor="nw")
        self._build_canvas_overlay()

        self.progress = tk.Canvas(main, width=self.cw + 2, height=px(22), bg=BG, highlightthickness=0)
        self.progress.pack(pady=(px(8), 0))

        tk.Label(main, text="// CONSOLA", font=F(12), fg=GREEN_DIM, bg=BG, anchor="w").pack(fill="x", pady=(px(8), px(4)))
        self.console = tk.Text(main, height=6, width=1, bg=PANEL, fg=GREEN_MID, font=F(12), relief="flat",
                               highlightthickness=1, highlightbackground=GREEN_DARK, state="disabled",
                               padx=px(8), pady=px(6), wrap="word", cursor="arrow")
        self.console.pack(fill="x")
        for tag, color in (("info", GREEN_MID), ("ok", GREEN), ("warn", AMBER), ("err", RED), ("time", GREEN_DARK)):
            self.console.tag_configure(tag, foreground=color)

    def _section(self, parent, text, top=0):
        tk.Label(parent, text=text, font=F(12), fg=GREEN_DIM, bg=BG, anchor="w").pack(
            fill="x", pady=(px(top), px(8)))

    def _status_row(self, grid, row, key):
        tk.Label(grid, text=key, font=F(13), fg=TEXT_DIM, bg=BG, anchor="w", width=10).grid(
            row=row, column=0, sticky="w", pady=px(2))
        value = tk.Label(grid, text="", font=F(13, True), fg=GREEN, bg=BG, anchor="w")
        value.grid(row=row, column=1, sticky="w")
        return value

    def _build_canvas_overlay(self):
        c, w, h, a = self.canvas, self.cw, self.ch, px(26)
        m = px(6)
        for (x, y, dx, dy) in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
            c.create_line(x, y, x + dx * a, y, fill=GREEN, width=2, tags="deco")
            c.create_line(x, y, x, y + dy * a, fill=GREEN, width=2, tags="deco")
        # etiqueta de modo (arriba a la izquierda), con sombra para que se lea sobre el video
        self.hud_shadow = c.create_text(px(19), px(17), anchor="nw", text="", fill="black", font=F(15, True))
        self.hud = c.create_text(px(18), px(16), anchor="nw", text="", fill=GREEN, font=F(15, True))
        # panel central de espera
        cx, cy, bw, bh = w // 2, h // 2, px(380), px(96)
        c.create_rectangle(cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2,
                           fill=BG, outline=GREEN_DIM, width=1, tags="idle")
        self.idle_title = c.create_text(cx, cy - px(16), text="", fill=GREEN, font=F(20, True), tags="idle")
        self.idle_sub = c.create_text(cx, cy + px(20), text="", fill=TEXT_DIM, font=F(13), tags="idle")
        self.idle_title_text, self.idle_sub_text = "SISTEMA EN ESPERA", "Seleccione una operación"

    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        x = max(0, (self.winfo_screenwidth() - w) // 2)
        y = max(0, (self.winfo_screenheight() - h) // 2 - px(20))
        self.geometry(f"+{x}+{y}")
        self.minsize(w, h)

    # ------------------------------------------------------------- consola
    def log(self, text, tag="info"):
        self.console.config(state="normal")
        self.console.insert("end", time.strftime("[%H:%M:%S] "), "time")
        self.console.insert("end", text + "\n", tag)
        self.console.see("end")
        self.console.config(state="disabled")

    def _poll_queue(self):
        try:
            while True:
                kind, *data = self.q.get_nowait()
                if kind == "log":
                    line = data[0]
                    tag = "err" if line.startswith("Error") else "warn" if line.startswith("Aviso") \
                        else "ok" if line.startswith("Entrenamiento completo") else "info"
                    self.log(line, tag)
                elif kind == "train_done":
                    self._on_train_done(data[0])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # -------------------------------------------------------- estado/datos
    def _scan(self):
        """Devuelve [(nombre, nº de muestras)], fecha de la muestra más reciente."""
        people, latest = [], 0.0
        data_dir = capture_faces.DATA_DIR
        if data_dir.exists():
            for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
                count = 0
                for f in folder.iterdir():
                    if f.suffix.lower() in train_model.EXTENSIONS:
                        count += 1
                        try:
                            latest = max(latest, f.stat().st_mtime)
                        except OSError:
                            pass
                if count:
                    people.append((folder.name, count))
        return people, latest

    def refresh_status(self, initial=False):
        try:
            people, latest = self._scan()
        except OSError as exc:
            self.log(f"Error leyendo data/: {exc}", "err")
            people, latest = [], 0.0

        self.people_box.config(state="normal")
        self.people_box.delete("1.0", "end")
        if people:
            for name, count in people:
                self.people_box.insert("end", f"{name.replace('_', ' ')[:24]:<24}", "name")
                self.people_box.insert("end", f" {count:>3}\n", "count")
        else:
            self.people_box.insert("end", "Sin registros todavía.\nUse 01 REGISTRAR ROSTRO.", "empty")
        self.people_box.config(state="disabled")
        self.lbl_people.config(text=str(len(people)), fg=GREEN if people else TEXT_DIM)

        model, labels = train_model.MODEL_PATH, train_model.LABELS_PATH
        need_train = False
        if not people:
            text, color = "SIN DATOS", TEXT_DIM
        elif not model.exists() or not labels.exists():
            text, color, need_train = "SIN ENTRENAR", AMBER, True
        else:
            try:
                outdated = latest > model.stat().st_mtime
            except OSError:
                outdated = True
            text, color, need_train = ("DESACTUALIZADO", AMBER, True) if outdated else ("LISTO", GREEN, False)
        self.lbl_model.config(text=text, fg=color)
        self.btn_train.set_attention(need_train and self.mode == IDLE)
        if initial and need_train:
            self.log("El modelo necesita entrenarse: use 02 ENTRENAR MODELO.", "warn")

    def _set_mode(self, mode):
        self.mode = mode
        idle = mode == IDLE
        for b in (self.btn_register, self.btn_train, self.btn_recog):
            b.set_enabled(idle)
        self.btn_stop.set_enabled(mode in (CAPTURE, RECOGNIZE))
        if not idle:
            self.btn_train.set_attention(False)
        self.canvas.itemconfig("idle", state="normal" if mode in (IDLE, TRAIN) else "hidden")
        cam_on = mode in (CAPTURE, RECOGNIZE)
        self.lbl_cam.config(text="ACTIVA" if cam_on else "EN ESPERA", fg=GREEN if cam_on else TEXT_DIM)
        if mode == TRAIN:
            self.idle_title_text, self.idle_sub_text = "ENTRENANDO MODELO", "Procesando muestras..."
        else:
            self.idle_title_text, self.idle_sub_text = "SISTEMA EN ESPERA", "Seleccione una operación"
        self._set_hud({IDLE: "", TRAIN: "", CAPTURE: "● REGISTRANDO", RECOGNIZE: "● VALIDANDO"}[mode])
        if mode != CAPTURE:
            self.set_progress(0, "—")

    def _set_hud(self, text):
        self.canvas.itemconfig(self.hud, text=text)
        self.canvas.itemconfig(self.hud_shadow, text=text)

    def set_progress(self, ratio, text):
        self.progress_value, self.progress_text = ratio, text
        c, w, h = self.progress, self.cw + 2, px(22)
        c.delete("all")
        c.create_rectangle(0, 0, w - 1, h - 1, outline=GREEN_DARK)
        if ratio > 0:
            c.create_rectangle(2, 2, 2 + (w - 5) * min(ratio, 1.0), h - 3, fill=GREEN_DIM, outline="")
        c.create_text(w // 2, h // 2, text=text, fill=WHITEGREEN if ratio > 0 else TEXT_DIM, font=F(12, True))

    # ------------------------------------------------------------- cámara
    def _open_camera(self):
        self.log("Abriendo cámara...")
        self.update_idletasks()
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
        if not cam.isOpened():
            cam.release()
            cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            cam.release()
            self.log("Error: no se pudo abrir la cámara. Verifique que esté conectada y libre "
                     "(cierre Teams, Zoom u otras apps).", "err")
            return False
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cam = cam
        return True

    def _close_camera(self):
        if self.cam is not None:
            self.cam.release()
            self.cam = None

    def _read_frame(self):
        ok, frame = self.cam.read()
        if not ok:
            self.log("Error: no se pudo leer un frame de la cámara.", "err")
            self._stop_mode()
            return None
        return frame

    def _detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        return gray, faces

    # ----------------------------------------------------------- acciones
    def on_register(self):
        if self.mode != IDLE:
            return
        if self.detector.empty():
            self.log("Error: no se pudo cargar el Haar Cascade.", "err")
            return
        dialog = NameDialog(self)
        self.wait_window(dialog)
        if not dialog.result:
            return
        name = capture_faces.clean_name(dialog.result)
        person_dir = capture_faces.DATA_DIR / name
        count = len(list(person_dir.glob("*.jpg")))
        if count >= capture_faces.MAX_SAMPLES:
            self.log(f"'{name}' ya tiene {count} muestras. Borre data/{name} para recapturar.", "warn")
            return
        if not self._open_camera():
            return
        self.cap = dict(name=name, dir=person_dir, count=count, saved=0, frame_id=0)
        self.log(f"Capturando para '{name}'. Mire a la cámara y mueva un poco la cabeza.", "ok")
        self._set_mode(CAPTURE)
        self.set_progress(count / capture_faces.MAX_SAMPLES, f"{name.replace('_', ' ')}  {count}/{capture_faces.MAX_SAMPLES}")

    def on_train(self):
        if self.mode != IDLE:
            return
        self._set_mode(TRAIN)
        self.log("Iniciando entrenamiento...")

        def work():
            writer = _QueueWriter(self.q)
            try:
                with contextlib.redirect_stdout(writer):
                    code = train_model.main()
            except Exception as exc:                      # no debe tumbar la interfaz
                self.q.put(("log", f"Error durante el entrenamiento: {exc}"))
                code = 1
            self.q.put(("train_done", code))

        threading.Thread(target=work, daemon=True).start()

    def _on_train_done(self, code):
        self._set_mode(IDLE)
        self.refresh_status()
        if code == 0:
            self.log("Modelo listo. Ya puede usar 03 VALIDAR RECONOCIMIENTO.", "ok")
        else:
            self.log("El entrenamiento no se completó. Revise los mensajes anteriores.", "err")

    def on_recognize(self):
        if self.mode != IDLE:
            return
        if not hasattr(cv2, "face"):
            self.log("Error: falta cv2.face. Instale opencv-contrib-python (y desinstale opencv-python).", "err")
            return
        if self.detector.empty():
            self.log("Error: no se pudo cargar el Haar Cascade.", "err")
            return
        if not train_model.MODEL_PATH.exists() or not train_model.LABELS_PATH.exists():
            self.log("Error: no existe el modelo entrenado. Use 02 ENTRENAR MODELO primero.", "err")
            self.btn_train.set_attention(True)
            return
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(str(train_model.MODEL_PATH))
            with open(train_model.LABELS_PATH, encoding="utf-8") as f:
                labels = {int(k): v for k, v in json.load(f).items()}
        except (cv2.error, OSError, ValueError) as exc:
            self.log(f"Error cargando el modelo: {exc}. Vuelva a entrenarlo.", "err")
            return
        if not self._open_camera():
            return
        self.recognizer, self.labels = recognizer, labels
        self.log("Validando reconocimiento... presione DETENER o Esc para salir.", "ok")
        self._set_mode(RECOGNIZE)

    def on_stop(self):
        if self.mode in (CAPTURE, RECOGNIZE):
            self._stop_mode()

    def _stop_mode(self):
        prev = self.mode
        self._close_camera()
        self._set_mode(IDLE)
        if prev == CAPTURE:
            cap = self.cap
            if cap["saved"]:
                done = cap["count"] >= capture_faces.MAX_SAMPLES
                self.log(f"Captura {'completa' if done else 'detenida'}: {cap['saved']} muestras nuevas de "
                         f"'{cap['name']}' ({cap['count']}/{capture_faces.MAX_SAMPLES}).", "ok")
                self.log("Ahora use 02 ENTRENAR MODELO para incluir a esta persona.", "warn")
            else:
                self.log("Captura detenida sin guardar muestras.", "warn")
            self.refresh_status()
        elif prev == RECOGNIZE:
            self.log("Validación detenida.")

    # ---------------------------------------------------- bucle principal
    def _loop(self):
        delay = 40
        frame = None
        try:
            if self.mode == CAPTURE:
                frame, delay = self._step_capture(), 5
            elif self.mode == RECOGNIZE:
                frame, delay = self._step_recognize(), 5
            else:
                frame = self.rain.step()
                dots = "_" if int(time.time() * 2) % 2 == 0 else " "
                self.canvas.itemconfig(self.idle_title, text=self.idle_title_text + dots)
                self.canvas.itemconfig(self.idle_sub, text=self.idle_sub_text)
            if frame is not None:
                self._show(frame)
        except Exception as exc:
            self.log(f"Error inesperado: {exc!r}", "err")
            self._stop_mode()
        now = time.strftime("%H:%M:%S")
        if now != self._last_clock:
            self._last_clock = now
            self.clock.config(text=now)
        self.after(delay, self._loop)

    def _show(self, frame):
        frame = fit_to_canvas(frame, self.cw, self.ch)
        ok, buf = cv2.imencode(".ppm", frame)
        if ok:
            self._photo = tk.PhotoImage(data=buf.tobytes())
            self.canvas.itemconfig(self.img_item, image=self._photo)

    def _step_capture(self):
        frame = self._read_frame()
        if frame is None:
            return None
        cap = self.cap
        gray, faces = self._detect(frame)
        hint = "Mire a la cámara y mueva un poco la cabeza"
        # Solo se guarda con exactamente un rostro, para no mezclar a otras personas
        if len(faces) == 1:
            x, y, w, h = faces[0]
            draw_face_box(frame, x, y, w, h, (65, 255, 0))
            cap["frame_id"] += 1
            hint = "Rostro detectado"
            if cap["frame_id"] % capture_faces.FRAME_STEP == 0:
                face = cv2.resize(gray[y:y + h, x:x + w], capture_faces.FACE_SIZE)
                cap["dir"].mkdir(parents=True, exist_ok=True)
                cap["count"] += 1
                cap["saved"] += 1
                if not cv2.imwrite(str(cap["dir"] / f"{cap['count']:03d}.jpg"), face):
                    self.log("Error: no se pudo guardar la muestra en disco.", "err")
                    self._stop_mode()
                    return frame
        elif len(faces) > 1:
            hint = "Se detectaron varios rostros: debe haber solo uno"
        else:
            hint = "No se detecta ningún rostro"
        total = capture_faces.MAX_SAMPLES
        self.set_progress(cap["count"] / total, f"{cap['name'].replace('_', ' ')}  {cap['count']}/{total}  ·  {hint}")
        if cap["count"] >= total:
            self._stop_mode()
        return frame

    def _step_recognize(self):
        frame = self._read_frame()
        if frame is None:
            return None
        gray, faces = self._detect(frame)
        for x, y, w, h in faces:
            face = cv2.resize(gray[y:y + h, x:x + w], recognize_faces.FACE_SIZE)
            label, distance = self.recognizer.predict(face)
            if distance < recognize_faces.THRESHOLD and label in self.labels:
                draw_face_box(frame, x, y, w, h, (65, 255, 0), self.labels[label].replace("_", " "),
                              f"Confianza: {distance:.1f}")
            else:
                draw_face_box(frame, x, y, w, h, (59, 59, 255), "Desconocido", f"Confianza: {distance:.1f}")
        self._set_hud(f"● VALIDANDO  ·  rostros: {len(faces)}")
        return frame

    def _on_close(self):
        self._close_camera()
        self.destroy()


def main() -> int:
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
