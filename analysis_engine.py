"""Compute spatial explanations using the Trainer checkpoint and preprocessing."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from collections import defaultdict
from concurrent.futures import CancelledError


PROJECT = Path(__file__).resolve().parent
TRAINER = PROJECT.parent / "BurnerEye Trainer"
if not (TRAINER / "burnereye").is_dir():
    raise ImportError(f"Не найден исходный код BurnerEye Trainer: {TRAINER}")
sys.path.insert(0, str(TRAINER))

from burnereye.video import find_video_files, parse_label_from_filename, parse_dataset_split_from_filename

VIDEO_EXTENSIONS = (".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v")
MASKS = (
    ("blur_small", "Размытие", 48, 32, 24, 16),
    ("black_small", "Чёрная маска", 48, 32, 24, 16),
    ("blur_large", "Размытие", 96, 56, 48, 28),
)


def window_count(count):
    ending = "окон"
    if not 11 <= count % 100 <= 14:
        ending = {1: "окно", 2: "окна", 3: "окна", 4: "окна"}.get(count % 10, ending)
    return f"{count} {ending}"


@dataclass(frozen=True)
class TestVideo:
    path: Path
    regime: str
    steam: int
    fuel: int
    fuel_code: str


@dataclass
class ModeMaps:
    video: TestVideo
    rgb: object
    cam: object
    abs_delta: dict
    signed_delta: dict
    scores: object
    windows: list


@dataclass
class Analysis:
    modes: list[ModeMaps]
    metadata: dict
    heads: tuple[str, ...]
    mask_labels: dict[str, str]


def discover_videos(root):
    candidates = defaultdict(list)
    all_modes, unidentified = set(), []
    for path in find_video_files(root, VIDEO_EXTENSIONS):
        relative = path.relative_to(root)
        split, error = parse_dataset_split_from_filename(path)
        if error == "missing_split_marker":
            folders = {name.casefold() for name in relative.parts[:-1]}
            split_folders = folders & {"train", "val", "test"}
            if len(split_folders) == 1:
                split = next(iter(split_folders))
        label = parse_label_from_filename(path.name.upper())
        if label.steam_g_h is None:
            for folder in reversed(relative.parts[:-1]):
                label = parse_label_from_filename(folder.upper())
                if label.steam_g_h is not None:
                    break
        if label.steam_g_h is None:
            if split == "test" or error == "ambiguous_split_marker":
                unidentified.append(str(relative))
            continue
        all_modes.add(label.regime_id)
        if split == "test":
            candidates[label.regime_id].append(TestVideo(
                path, label.regime_id, label.steam_g_h, label.fuel_g_h, label.fuel_code
            ))
        elif error == "ambiguous_split_marker":
            unidentified.append(str(relative))

    def sort_key(regime):
        label = parse_label_from_filename(regime.replace("_", ""))
        return label.steam_g_h, label.fuel_code, label.fuel_g_h

    return (
        {key: candidates[key] for key in sorted(candidates, key=sort_key)},
        sorted(all_modes - candidates.keys(), key=sort_key),
        unidentified,
    )


def locate_checkpoint(folder):
    for path in (folder / "model" / "best_model.pt", folder / "best_model.pt"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Не найден model/best_model.pt или best_model.pt в папке {folder}")


def read_test_manifest(checkpoint):
    for parent in (checkpoint.parent, checkpoint.parent.parent):
        path = parent / "data" / "test_split.csv"
        if path.is_file():
            with path.open(encoding="utf-8-sig", newline="") as handle:
                return {Path(row["path"]).resolve(): row for row in csv.DictReader(handle)}
    return {}


def choose_windows(video, count, config, manifest):
    import numpy as np
    from burnereye.dataset_prep import estimate_num_clips
    from burnereye.input_modes import sample_timing
    from burnereye.video import probe_video

    path = video.path.resolve()
    row = manifest.get(path)
    if row and row.get("fps") and row.get("num_clips"):
        fps, clips = float(row["fps"]), int(row["num_clips"])
    else:
        info = probe_video(path)
        if not info.opened or not info.fps:
            raise ValueError(f"Не удалось прочитать FPS и длину видео {video.path.name}")
        fps = info.fps
        clips = estimate_num_clips(info.duration_s, config, fps, info.frame_count)
    if clips <= 0:
        raise ValueError(f"В видео {video.path.name} нет полного окна для выбранной модели.")
    duration, stride = sample_timing(config, fps)
    starts = []
    for indices in np.array_split(np.arange(clips), min(count, clips)):
        clip = int(indices[len(indices) // 2])
        start = clip * stride
        starts.append((clip, start, start + duration))
    return starts


def read_input(path, start, config, cache, shape):
    import numpy as np
    from burnereye.frame_cache import _canonical
    from burnereye.preprocessing import sample_clip

    if cache is not None:
        stat = path.stat()
        for timestamp in (start, float(format(start, ".12g"))):
            key = hashlib.sha256(_canonical({
                "path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "settings": cache.settings, "start_s": timestamp, "format": "normalized-float32-v1",
            })).hexdigest()
            candidate = cache.root / "normalized" / f"{key}.npy"
            if candidate.is_file():
                try:
                    frames = np.load(candidate, allow_pickle=False)
                except (OSError, ValueError, EOFError):
                    continue
                if frames.shape == shape and frames.dtype == np.float32 and np.isfinite(frames).all():
                    return frames, "cache"
    return sample_clip(str(path), start, config), "video"


def mask_grid(height, width):
    result, labels = {}, {}
    for name, fill, w, h, dx, dy in MASKS:
        w, h = min(width, max(1, round(w * width / 480))), min(height, max(1, round(h * height / 112)))
        dx, dy = max(1, round(dx * width / 480)), max(1, round(dy * height / 112))
        xs = sorted(set(range(0, width - w + 1, dx)) | {width - w})
        ys = sorted(set(range(0, height - h + 1, dy)) | {height - h})
        result[name] = [(x, y, x + w, y + h) for y in ys for x in xs]
        labels[name] = f"{fill} {w}×{h}"
    return result, labels


def gradcam(model, frames, predict, heads):
    import numpy as np
    import torch

    activations = []
    capture = model.backbone.layer4[-1].register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    try:
        # cuDNN has no eval-mode GRU backward. Keep eval semantics and use native ops.
        with torch.backends.cudnn.flags(enabled=model.average_image and torch.backends.cudnn.enabled):
            scores = predict(frames[None].detach().requires_grad_())
            activation, = activations
            maps = []
            for i in range(len(heads)):
                gradient, = torch.autograd.grad(scores[0, i], activation, retain_graph=i + 1 < len(heads))
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                maps.append((weights * activation).sum(dim=1).detach().cpu().numpy())
    finally:
        capture.remove()
    raw = np.stack(maps)
    both_signs = np.maximum(np.stack((raw, -raw)), 0).mean(axis=2)
    peaks = both_signs.max(axis=(-2, -1), keepdims=True)
    normalized = np.divide(both_signs, peaks, out=np.zeros_like(both_signs), where=peaks > 0)
    return scores[0].detach(), normalized


def occlusion(frames, predict, original, replacement, boxes, batch_size, check_cancel):
    import numpy as np
    import torch

    delta = []
    with torch.inference_mode():
        for start in range(0, len(boxes), batch_size):
            check_cancel()
            chunk = boxes[start:start + batch_size]
            variants = frames.repeat(len(chunk), 1, 1, 1, 1)
            for i, (x0, y0, x1, y1) in enumerate(chunk):
                variants[i, ..., y0:y1, x0:x1] = replacement[..., y0:y1, x0:x1]
            delta.append((original - predict(variants)).cpu().numpy())
    return np.concatenate(delta)


def paint_boxes(shape, boxes, values):
    import numpy as np

    image = np.zeros((values.shape[1], *shape), dtype=np.float32)
    coverage = np.zeros(shape, dtype=np.uint16)
    for (x0, y0, x1, y1), value in zip(boxes, values, strict=True):
        image[:, y0:y1, x0:x1] += value[:, None, None]
        coverage[y0:y1, x0:x1] += 1
    return image / coverage


def calculate(checkpoint, videos, count, cancel, progress):
    import cv2
    import numpy as np
    import torch
    from burnereye.config import resolve_input_hw
    from burnereye.frame_cache import PreprocessedFrameCache
    from burnereye.input_modes import model_frame_count
    from burnereye.model import build_model

    def check_cancel():
        if cancel.is_set():
            raise CancelledError()

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    package = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = package["config"]
    config["model"]["pretrained"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, package.get("mappings")).to(device).eval()
    model.load_state_dict(package["model_state_dict"], strict=True)
    model.requires_grad_(False)
    heads = ("steam", "fuel") if model.output_mode == "multi" else (model.output_mode,)
    norms = [package["normalization"][f"{head}_g_h"] for head in heads]
    score_mean = torch.tensor([norm["mean"] for norm in norms], device=device)
    score_std = torch.tensor([norm["std"] for norm in norms], device=device)

    def predict(frames):
        outputs = model(frames)
        return torch.cat([outputs[head] for head in heads], dim=1) * score_std + score_mean

    rgb_mean = np.asarray(config["normalize"]["mean"], dtype=np.float32)[None, :, None, None]
    rgb_std = np.asarray(config["normalize"]["std"], dtype=np.float32)[None, :, None, None]
    black = torch.as_tensor(-rgb_mean / rgb_std, device=device)
    height, width = resolve_input_hw(config["video"]["input_size"])
    shape = (model_frame_count(config), 3, height, width)
    sigma = max(1, 12 * height / 112)
    batch_size = max(1, min(24, (24 * 112 * 480) // (shape[0] * height * width)))
    boxes, mask_labels = mask_grid(height, width)
    data_config = config.get("data", {})
    cache_path = Path(data_config.get("frame_cache_dir") or checkpoint.parent.parent / "cache" / "preprocessed_frames")
    cache = PreprocessedFrameCache(cache_path, config) if (cache_path / "normalized").is_dir() else None
    manifest = read_test_manifest(checkpoint)
    tasks = []
    for video in videos:
        check_cancel()
        tasks.append((video, choose_windows(video, count, config, manifest)))
    total = sum(len(windows) for _, windows in tasks)
    completed, results = 0, []
    for video, windows in tasks:
        rgb, cams, predictions, records = [], [], [], []
        effects = defaultdict(list)
        for index, start, end in windows:
            check_cancel()
            progress(completed, total, f"{video.regime.replace('_', '')} · окно {len(records) + 1} из {len(windows)} · {device}")
            try:
                array, source = read_input(video.path, start, config, cache, shape)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(f"{video.path.name}, {start:.3f} с — {exc}") from exc
            frames = torch.from_numpy(array).to(device)
            score, cam = gradcam(model, frames, predict, heads)
            # Use the same mask for every model input frame when the model uses a GRU.
            blur = np.stack([
                cv2.GaussianBlur(frame.transpose(1, 2, 0), (0, 0), sigma).transpose(2, 0, 1)
                for frame in array
            ])
            blurry = torch.from_numpy(np.ascontiguousarray(blur)).to(device)
            for name, positions in boxes.items():
                replacement = black.expand_as(frames) if name == "black_small" else blurry
                effects[name].append(occlusion(frames, predict, score, replacement, positions, batch_size, check_cancel))
            original = np.clip((array * rgb_std + rgb_mean).mean(axis=0), 0, 1)
            rgb.append(original.transpose(1, 2, 0))
            cams.append(cam)
            prediction = score.cpu().numpy()
            predictions.append(prediction)
            records.append({"index": index, "start_s": start, "end_s": end, "source": source,
                            "prediction_g_h": prediction.tolist()})
            completed += 1
        result = ModeMaps(
            video, np.mean(rgb, axis=0), np.mean(cams, axis=0),
            {name: paint_boxes((height, width), boxes[name], np.abs(values).mean(axis=0)) for name, values in effects.items()},
            {name: paint_boxes((height, width), boxes[name], np.mean(values, axis=0)) for name, values in effects.items()},
            np.mean(predictions, axis=0), records,
        )
        if not all(np.isfinite(values).all() for values in (*result.abs_delta.values(), result.cam, result.scores)):
            raise ValueError(f"Модель вернула некорректные значения для {video.path.name}.")
        results.append(result)
    metadata = {
        "checkpoint": str(checkpoint), "epoch": package.get("epoch"), "device": str(device),
        "input_mode": "average8" if model.average_image else "sequence",
        "input_shape": list(shape), "roi": config.get("roi", {}),
        "cam_layer": "backbone.layer4[-1]", "heads": list(heads),
        "mask_labels": mask_labels, "mask_boxes_xyxy": boxes, "blur_sigma_px": sigma,
        "delta": "unmasked_prediction_g_h - masked_prediction_g_h",
        "videos": [{"regime": mode.video.regime, "path": str(mode.video.path),
                    "steam_g_h": mode.video.steam, "fuel_g_h": mode.video.fuel,
                    "windows": mode.windows} for mode in results],
    }
    return Analysis(results, metadata, heads, mask_labels)
