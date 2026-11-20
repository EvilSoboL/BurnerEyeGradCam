"""Desktop window for comparing Grad-CAM and occlusion across test videos."""
from __future__ import annotations

from concurrent.futures import CancelledError
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import traceback

from analysis_engine import PROJECT, calculate, discover_videos, locate_checkpoint, window_count
from plots import export

DESKTOP = Path.home() / "Desktop"
DEFAULT_VIDEO = DESKTOP / "Обучающие видео" / "Diesel-Steam-Удалены 2 режима и непохожий train"
DEFAULT_MODEL = DESKTOP / "Модели предсказания расходов" / (
    "Две ветви GRU, усредненная картинка, дизель, удален непохожий train, dropout, unfreeze15"
)


class GradCamApp:
    def __init__(self, root, videos=DEFAULT_VIDEO, model=DEFAULT_MODEL, output=PROJECT / "results"):
        self.root = root
        root.title("BurnerEye Grad-CAM")
        root.geometry("1320x850")
        root.minsize(850, 620)
        root.configure(background="#edf2f8")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.video_path = tk.StringVar(value=str(videos))
        self.model_path = tk.StringVar(value=str(model))
        self.output_path = tk.StringVar(value=str(output))
        self.count = tk.StringVar(value="12")
        self.view = tk.StringVar()
        self.status = tk.StringVar(value="Выберите папку с видео и моделью.")
        self.scan_status = tk.StringVar()
        self.export_path = tk.StringVar()
        self.candidates = {}
        self.selected = {}
        self.scanned_root = None
        self.messages = Queue()
        self.cancel = Event()
        self.busy = False
        self.closing = False
        self.figures = {}
        self.destination = None
        self.source_image = None
        self.photo = None
        self.resize_job = None
        self.editable = []
        self._layout()
        if videos.is_dir():
            root.after(50, self.scan)
        root.after(65, self.poll)

    def _layout(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#edf2f8")
        style.configure("TLabel", background="#edf2f8", foreground="#172840", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Quiet.TLabel", foreground="#43556d")
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 5))
        style.configure("Run.TButton", background="#185ab8", foreground="#fff")
        style.map("Run.TButton", background=[("disabled", "#8eabc9"), ("active", "#124891")])
        style.configure("Treeview", rowheight=25, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        frame = ttk.Frame(self.root, padding=(17, 11, 17, 12))
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)
        ttk.Label(frame, text="Grad-CAM и окклюзия", style="Title.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        paths = ttk.Frame(frame)
        paths.grid(row=1, column=0, sticky="ew")
        paths.columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate((
            ("Папка с видео", self.video_path),
            ("Папка модели", self.model_path),
            ("Куда сохранить", self.output_path),
        )):
            ttk.Label(paths, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            entry = ttk.Entry(paths, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            browse = ttk.Button(paths, text="Обзор…", command=lambda var=variable: self.browse(var))
            browse.grid(row=row, column=2, padx=(7, 0), pady=2)
            self.editable.extend((entry, browse))
        self.video_path.trace_add("write", self.invalidate_scan)
        tools = ttk.Frame(frame)
        tools.grid(row=2, column=0, sticky="ew", pady=(9, 8))
        scan_button = ttk.Button(tools, text="Найти тестовые видео", command=self.scan)
        scan_button.pack(side="left", padx=(0, 14))
        self.editable.append(scan_button)
        ttk.Label(tools, text="Окон на видео").pack(side="left", padx=(0, 6))
        count = ttk.Spinbox(tools, from_=1, to=50, textvariable=self.count, width=5)
        count.pack(side="left", padx=(0, 14))
        self.editable.append(count)
        self.run_button = ttk.Button(tools, text="Построить и сохранить", style="Run.TButton", command=self.start)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(tools, text="Отменить", command=self.cancel_run, state="disabled")
        self.cancel_button.pack(side="left", padx=7)
        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.grid(row=3, column=0, sticky="ew")
        ttk.Label(frame, textvariable=self.status, style="Quiet.TLabel").grid(row=4, column=0, sticky="ew", pady=(4, 8))

        body = ttk.Panedwindow(frame, orient=tk.HORIZONTAL)
        body.grid(row=5, column=0, sticky="nsew")
        sources = ttk.Frame(body, padding=(0, 4, 8, 0), width=315)
        results = ttk.Frame(body, padding=(8, 4, 0, 0))
        body.add(sources, weight=0)
        body.add(results, weight=1)
        sources.columnconfigure(0, weight=1)
        sources.rowconfigure(1, weight=1)
        results.columnconfigure(0, weight=1)
        results.rowconfigure(1, weight=1)
        ttk.Label(sources, text="Тестовые видео", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.tree = ttk.Treeview(sources, columns=("regime", "video"), show="headings", selectmode="browse")
        self.tree.heading("regime", text="Режим")
        self.tree.heading("video", text="Файл")
        self.tree.column("regime", width=100, minwidth=84, stretch=False)
        self.tree.column("video", width=182, minwidth=106)
        self.tree.grid(row=1, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(sources, orient="vertical", command=self.tree.yview)
        tree_scroll.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self.show_alternatives)
        self.alternative = ttk.Combobox(sources, state="disabled", width=24)
        self.alternative.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.alternative.bind("<<ComboboxSelected>>", self.choose_alternative)
        ttk.Label(sources, textvariable=self.scan_status, style="Quiet.TLabel", wraplength=295,
                  justify="left").grid(row=3, column=0, sticky="w", pady=(6, 0))
        header = ttk.Frame(results)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(header, text="Карты всех режимов", style="Section.TLabel").pack(side="left", padx=(0, 10))
        self.view_combo = ttk.Combobox(header, textvariable=self.view, state="disabled", width=21)
        self.view_combo.pack(side="left")
        self.view_combo.bind("<<ComboboxSelected>>", self.show_figure)
        self.open_button = ttk.Button(header, text="Открыть папку", command=self.open_results, state="disabled")
        self.open_button.pack(side="right")
        self.canvas = tk.Canvas(results, background="#f4f7fb", borderwidth=0, highlightthickness=1,
                                highlightbackground="#c8d4e1")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        result_scroll = ttk.Scrollbar(results, orient="vertical", command=self.canvas.yview)
        result_scroll.grid(row=1, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=result_scroll.set)
        self.canvas.bind("<Configure>", self.queue_resize)
        self.canvas.bind("<MouseWheel>", lambda event: self.canvas.yview_scroll(-int(event.delta / 120), "units"))
        self.empty_text = self.canvas.create_text(24, 24, anchor="nw",
            text="После расчёта здесь появятся карты для найденных режимов.",
            font=("Segoe UI", 11), fill="#43556d")
        ttk.Label(results, textvariable=self.export_path, style="Quiet.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(5, 0)
        )

    def browse(self, variable):
        current = Path(variable.get())
        selected = filedialog.askdirectory(parent=self.root, initialdir=str(current if current.is_dir() else current.parent))
        if selected:
            variable.set(selected)
            if variable is self.video_path:
                self.scan()

    def invalidate_scan(self, *_):
        self.scanned_root = None
        self.candidates.clear()
        self.selected.clear()
        self.tree.delete(*self.tree.get_children())
        self.show_alternatives()
        self.scan_status.set("")

    def scan(self):
        if self.busy:
            return False
        try:
            path = self.folder(self.video_path, "видео")
            candidates, missing, unidentified = discover_videos(path)
        except (OSError, ValueError) as exc:
            self.failure(str(exc))
            return False
        self.scanned_root = path
        self.candidates = candidates
        self.selected = {regime: options[0] for regime, options in candidates.items()}
        self.tree.delete(*self.tree.get_children())
        for regime, video in self.selected.items():
            self.tree.insert("", "end", iid=regime, values=(regime.replace("_", ""), video.path.name))
        for regime in missing:
            self.tree.insert("", "end", iid=regime, values=(regime.replace("_", ""), "Нет тестового видео"))
        if self.selected:
            first = next(iter(self.selected))
            self.tree.selection_set(first)
            self.tree.focus(first)
        else:
            self.show_alternatives()
        notes = [f"Режимов для расчёта — {len(self.selected)}"]
        if missing:
            notes.append(f"Без тестового видео — {len(missing)}")
        if unidentified:
            notes.append(f"Тестовых файлов с неясным режимом или разбиением — {len(unidentified)}")
        self.scan_status.set("\n".join(notes))
        self.status.set("Видео найдены." if self.selected else "Нужны тестовые файлы с именами вроде S400D600_test.mkv.")
        return bool(self.selected)

    def show_alternatives(self, _event=None):
        selection = self.tree.selection()
        key = selection[0] if selection else ""
        options = self.candidates.get(key, [])
        if len(options) > 1 and not self.busy:
            self.alternative.configure(state="readonly", values=[str(v.path.relative_to(self.scanned_root)) for v in options])
            self.alternative.current(options.index(self.selected[key]))
        else:
            self.alternative.configure(state="disabled", values=[])
            self.alternative.set("")

    def choose_alternative(self, _event):
        key, = self.tree.selection()
        self.selected[key] = self.candidates[key][self.alternative.current()]
        self.tree.set(key, "video", self.selected[key].path.name)

    @staticmethod
    def folder(variable, description, require_exists=True):
        value = variable.get().strip().strip('"')
        if not value:
            raise ValueError(f"Укажите папку для {description}.")
        path = Path(value).expanduser().resolve()
        if require_exists and not path.is_dir():
            raise ValueError(f"Папка не найдена.\n{path}")
        return path

    def start(self):
        if self.busy:
            return
        try:
            model = self.folder(self.model_path, "модели")
            checkpoint = locate_checkpoint(model)
            if model.name.casefold() == "model" and checkpoint.parent == model:
                model = model.parent
            output = self.folder(self.output_path, "сохранения", require_exists=False)
            target = (output / model.name).resolve()
            if target.is_relative_to(model):
                raise ValueError("Выберите папку сохранения вне исходной папки модели.")
            if target.exists() and not target.is_dir():
                raise ValueError(f"На месте папки результатов уже существует файл.\n{target}")
            count = int(self.count.get())
            if not 1 <= count <= 50:
                raise ValueError("Укажите от 1 до 50 окон на видео.")
            if self.scanned_root != self.folder(self.video_path, "видео") and not self.scan():
                return
            if not self.selected:
                raise ValueError("В выбранной папке нет тестовых видео с определённым режимом.")
            output.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            self.failure(str(exc))
            return
        self.cancel.clear()
        self.progress.configure(value=0)
        self.status.set("Загружаю модель и определяю окна видео…")
        self.set_busy(True)
        Thread(target=self.run, args=(checkpoint, model, list(self.selected.values()), output, count),
               daemon=True).start()

    def run(self, checkpoint, model, videos, output, count):
        def check_cancel():
            if self.cancel.is_set():
                raise CancelledError()

        try:
            analysis = calculate(checkpoint, videos, count, self.cancel,
                                 lambda done, total, status: self.messages.put(("progress", (done, total, status))))
            destination, figures = export(analysis, output, model, check_cancel,
                                          lambda status: self.messages.put(("progress", (1, 1, status))))
            self.messages.put(("success", (destination, figures, len(analysis.modes),
                                           sum(len(mode.windows) for mode in analysis.modes))))
        except CancelledError:
            self.messages.put(("cancelled", None))
        except Exception as exc:
            traceback.print_exc()
            self.messages.put(("error", str(exc)))

    def poll(self):
        try:
            while True:
                event, payload = self.messages.get_nowait()
                if event == "progress":
                    done, total, status = payload
                    self.progress.configure(maximum=total, value=done)
                    if not self.cancel.is_set():
                        self.status.set(status)
                    continue
                self.set_busy(False)
                if self.closing:
                    self.root.destroy()
                    return
                if event == "success":
                    self.destination, self.figures, modes, windows = payload
                    self.export_path.set(f"Сохранено в {self.destination}")
                    self.view_combo.configure(values=list(self.figures), state="readonly")
                    self.view_combo.current(0)
                    self.open_button.configure(state="normal")
                    self.show_figure()
                    self.status.set(f"Готово · режимов — {modes} · {window_count(windows)}")
                elif event == "cancelled":
                    self.status.set("Расчёт отменён.")
                else:
                    self.failure(payload)
        except Empty:
            pass
        self.root.after(65, self.poll)

    def set_busy(self, state):
        self.busy = state
        for control in self.editable + [self.run_button]:
            control.configure(state="disabled" if state else "normal")
        self.cancel_button.configure(state="normal" if state else "disabled")
        self.show_alternatives()

    def cancel_run(self):
        self.cancel.set()
        self.cancel_button.configure(state="disabled")
        self.status.set("Останавливаю расчёт…")

    def failure(self, message):
        self.status.set("Расчёт не выполнен.")
        messagebox.showerror("BurnerEye Grad-CAM", message, parent=self.root)

    def show_figure(self, _event=None):
        from PIL import Image

        filename = self.figures.get(self.view.get())
        if filename is not None:
            try:
                with Image.open(filename) as image:
                    self.source_image = image.convert("RGB")
            except OSError as exc:
                self.failure(str(exc))
                return
            self.render_figure(reset=True)

    def queue_resize(self, _event):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(75, self.render_figure)

    def render_figure(self, reset=False):
        from PIL import Image, ImageTk

        self.resize_job = None
        if self.source_image is None:
            return
        width = max(1, self.canvas.winfo_width() - 2)
        height = max(1, round(width * self.source_image.height / self.source_image.width))
        position = 0 if reset else self.canvas.yview()[0]
        self.photo = ImageTk.PhotoImage(self.source_image.resize((width, height), Image.Resampling.LANCZOS))
        self.canvas.delete("all")
        self.canvas.create_image(1, 1, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, width, height))
        self.canvas.yview_moveto(position)

    def open_results(self):
        if self.destination is not None:
            try:
                os.startfile(self.destination)
            except OSError as exc:
                self.failure(str(exc))

    def close(self):
        if self.busy:
            self.closing = True
            self.cancel_run()
        else:
            self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    GradCamApp(window)
    window.mainloop()
