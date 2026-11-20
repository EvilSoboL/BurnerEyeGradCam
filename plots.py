"""All-regime figures used by the desktop preview and exported as PNGs."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis_engine import window_count


def write_figures(analysis, destination, check_cancel, progress):
    import cv2
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    modes = analysis.modes
    heads = analysis.heads
    codes = {mode.video.fuel_code for mode in modes}
    fuel_label = {"D": "Дизель", "G": "Газ"}.get(next(iter(codes)), "Топливо") if len(codes) == 1 else "Топливо"
    head_names = {"steam": "Пар", "fuel": fuel_label}
    scales = {
        name: max(1, float(np.ceil(max(mode.abs_delta[name].max() for mode in modes))))
        for name in analysis.mask_labels
    }
    views = [
        ("Обзор", "overview", "Grad-CAM и окклюзия по всем режимам", "blur_small"),
        ("Grad-CAM", "gradcam", "Grad-CAM по всем режимам", None),
        *((label, name, f"Окклюзионная чувствительность · {label}", name)
          for name, label in analysis.mask_labels.items()),
    ]
    files = {}

    def overlay(rgb, values, palette, limit, signed=False):
        h, w = rgb.shape[:2]
        if values.shape != (h, w):
            values = cv2.resize(values, (w, h), interpolation=cv2.INTER_LINEAR)
        value = np.clip(values / limit, -1 if signed else 0, 1)
        tint = colormaps[palette]((value + 1) / 2 if signed else value)[..., :3]
        alpha = (0.87 * np.abs(value) ** 0.57)[..., None]
        return rgb * (1 - alpha) + tint * alpha

    def show(ax, rgb):
        ax.imshow(rgb, interpolation="nearest")
        ax.set_axis_off()

    def cam_columns(sign, label):
        return [(f"{head_names[head]}\n{label}", "magma", 1, False, lambda mode, h=h: mode.cam[sign, h])
                for h, head in enumerate(heads)]

    def occlusion_columns(mask, signed):
        label = "Среднее Δ" if signed else "Среднее |Δ|"
        return [(f"{head_names[head]}\n{label}", "RdBu_r" if signed else "inferno", scales[mask], signed,
                 lambda mode, h=h: (mode.signed_delta if signed else mode.abs_delta)[mask][h])
                for h, head in enumerate(heads)]

    for label, key, title, mask in views:
        check_cancel()
        progress(f"Сохраняю · {label}")
        if key == "overview":
            columns = cam_columns(0, "Grad-CAM +") + occlusion_columns(mask, False)
            caption = f"{analysis.mask_labels[mask]}. Среднее |Δ| показано в г/ч."
        elif key == "gradcam":
            columns = cam_columns(0, "Grad-CAM +") + cam_columns(1, "Grad-CAM −")
            caption = "Положительная и отрицательная карты нормированы отдельно в каждом окне."
        else:
            columns = occlusion_columns(mask, False) + occlusion_columns(mask, True)
            caption = "Δ = прогноз до маски − после маски, г/ч. Красный показывает снижение после маски, синий — повышение."
        row_count = len(modes)
        fig_width = 14 if len(heads) == 2 else 10
        h, w = modes[0].rgb.shape[:2]
        row_height = max(.97, min(2.6, .27 + .96 * fig_width / (len(columns) + 1) * h / w))
        fig_height = 1.8 + row_count * row_height
        fig = Figure(figsize=(fig_width, fig_height), facecolor="#f4f7fb")
        fig.text(.028, 1 - .27 / fig_height, title, va="top", fontsize=19, weight="bold", color="#13233b")
        fig.text(.028, 1 - .59 / fig_height, caption, va="top", fontsize=10, color="#43536a")
        grid = fig.add_gridspec(row_count, len(columns) + 1, left=.028, right=.978,
                               top=1 - 1.10 / fig_height, bottom=.68 / fig_height, wspace=.06, hspace=.47)
        for row, mode in enumerate(modes):
            check_cancel()
            regime = mode.video.regime.replace("_", "")
            original = fig.add_subplot(grid[row, 0])
            show(original, mode.rgb)
            if row == 0:
                original.set_title("Вход модели\nСреднее изображение", fontsize=9, color="#43536a")
            original.text(0, -.16, f"{regime}  ·  {window_count(len(mode.windows))}", transform=original.transAxes,
                          va="top", fontsize=9, color="#13233b", weight="bold")
            for col, (heading, palette, limit, signed, values) in enumerate(columns, start=1):
                ax = fig.add_subplot(grid[row, col])
                show(ax, overlay(mode.rgb, values(mode), palette, limit, signed))
                if row == 0:
                    ax.set_title(heading, fontsize=9, color="#43536a")
        legends = dict.fromkeys((palette, limit, signed) for _, palette, limit, signed, _ in columns)
        for i, (palette, limit, signed) in enumerate(legends):
            colors = Normalize(-limit if signed else 0, limit)
            ax = fig.add_axes([.10 + .47 * i, .32 / fig_height, .31, .12 / fig_height])
            bar = fig.colorbar(ScalarMappable(norm=colors, cmap=palette), cax=ax, orientation="horizontal")
            bar.set_ticks([-limit, 0, limit] if signed else [0, limit])
            bar.ax.tick_params(labelsize=8, colors="#43536a")
            bar.ax.set_title("Grad-CAM" if palette == "magma" else ("Δ, г/ч" if signed else "|Δ|, г/ч"),
                             fontsize=8, color="#43536a", pad=3)
        filename = f"{key}.png"
        fig.savefig(destination / filename, dpi=145, facecolor=fig.get_facecolor())
        fig.clear()
        files[label] = filename
    return files


def export(analysis, output, model_folder, check_cancel, progress):
    root = output / model_folder.name
    if root.resolve() == model_folder.resolve() or root.resolve().is_relative_to(model_folder.resolve()):
        raise ValueError("Выберите папку сохранения вне исходной папки модели.")
    with TemporaryDirectory(prefix=".gradcam-", dir=output) as directory:
        staged = Path(directory) / "figures"
        staged.mkdir()
        files = write_figures(analysis, staged, check_cancel, progress)
        (staged / "windows.json").write_text(
            json.dumps(analysis.metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        check_cancel()
        if root.is_dir():
            destination = root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        else:
            destination = root
        staged.rename(destination)
    return destination, {label: destination / filename for label, filename in files.items()}
