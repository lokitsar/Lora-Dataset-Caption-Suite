import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


ANALYSIS_SCHEMA_VERSION = 1


def analysis_config_version(config):
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _orientation(width, height):
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


def _basic_metrics(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    horizontal = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    vertical = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    return {
        "brightness_mean": round(float(gray.mean()), 3),
        "contrast_stddev": round(float(gray.std()), 3),
        "edge_sharpness": round(float(horizontal + vertical), 3),
    }


def _normalize_box(box, width, height):
    left, top, right, bottom = (float(value) for value in box)
    left = min(max(left, 0.0), float(width))
    top = min(max(top, 0.0), float(height))
    right = min(max(right, left), float(width))
    bottom = min(max(bottom, top), float(height))
    pixels = [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)]
    normalized = [
        round(left / width, 6),
        round(top / height, 6),
        round(right / width, 6),
        round(bottom / height, 6),
    ]
    return pixels, normalized


def _union_boxes(detections):
    boxes = [item["bbox"] for item in detections if item.get("bbox")]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _resolve_ultralytics_model(model_name):
    value = str(model_name or "").strip().replace("\\", "/")
    if not value or value.casefold() == "none":
        return None
    direct = Path(value).expanduser()
    if direct.is_file():
        return direct.resolve()

    try:
        import folder_paths
    except ImportError as error:
        raise FileNotFoundError(f"Cannot resolve Ultralytics model without ComfyUI: {value}") from error

    category_candidates = [("ultralytics", value)]
    if value.startswith("bbox/"):
        category_candidates.append(("ultralytics_bbox", value[5:]))
    elif value.startswith("segm/"):
        category_candidates.append(("ultralytics_segm", value[5:]))
    for category, filename in category_candidates:
        try:
            resolved = folder_paths.get_full_path(category, filename)
        except (KeyError, TypeError):
            resolved = None
        if resolved and Path(resolved).is_file():
            return Path(resolved).resolve()

    model_roots = set()
    for category in ("diffusion_models", "checkpoints", "vae", "text_encoders", "loras"):
        try:
            registered = folder_paths.get_folder_paths(category)
        except KeyError:
            continue
        for registered_path in registered:
            model_roots.add(Path(registered_path).resolve().parent)
    for root in sorted(model_roots, key=lambda path: str(path).casefold()):
        candidate = root / "ultralytics" / Path(value)
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Ultralytics model is not registered or present: {value}")


class AnalysisProvider(ABC):
    @abstractmethod
    def analyze(self, image_path, profile, context=None):
        raise NotImplementedError


class BasicAnalysisProvider(AnalysisProvider):
    def analyze(self, image_path, profile, context=None):
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            metrics = _basic_metrics(image)
            has_alpha = "A" in image.getbands()
        return {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "dimensions": {
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 6),
                "orientation": _orientation(width, height),
                "megapixels": round((width * height) / 1_000_000, 4),
                "has_alpha": has_alpha,
            },
            "quality": metrics,
            "subjects": [],
            "faces": [],
            "counts": {"objects": 0, "persons": 0, "faces": 0},
            "content_bounds": None,
            "provider": "basic",
        }


class UltralyticsAnalysisProvider(BasicAnalysisProvider):
    def __init__(self, config):
        self.config = dict(config)
        self.subject_model = None
        self.face_model = None

    def _load_model(self, field):
        configured = self.config.get(field, "none")
        if not configured or str(configured).casefold() == "none":
            return None
        attribute = "subject_model" if field == "subject_model" else "face_model"
        loaded = getattr(self, attribute)
        if loaded is None:
            from ultralytics import YOLO

            loaded = YOLO(str(_resolve_ultralytics_model(configured)))
            setattr(self, attribute, loaded)
        return loaded

    def _detect(self, model, image, kind):
        if model is None:
            return []
        results = model.predict(
            source=image,
            conf=float(self.config.get("confidence", 0.3)),
            imgsz=int(self.config.get("image_size", 640)),
            device=self.config.get("device", "cpu"),
            verbose=False,
        )
        detections = []
        if not results or results[0].boxes is None:
            return detections
        boxes = results[0].boxes
        names = results[0].names
        height, width = image.height, image.width
        for index in range(len(boxes)):
            label = str(names.get(int(boxes.cls[index].item()), kind))
            confidence = round(float(boxes.conf[index].item()), 6)
            pixels, normalized = _normalize_box(boxes.xyxy[index].tolist(), width, height)
            detections.append({
                "label": "face" if kind == "face" else label,
                "confidence": confidence,
                "bbox": pixels,
                "bbox_normalized": normalized,
            })
        return detections

    def analyze(self, image_path, profile, context=None):
        result = super().analyze(image_path, profile, context)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            subjects = self._detect(self._load_model("subject_model"), rgb, "object")
            faces = self._detect(self._load_model("face_model"), rgb, "face")
        result.update({
            "provider": "ultralytics",
            "subjects": subjects,
            "faces": faces,
            "counts": {
                "objects": len(subjects),
                "persons": sum(1 for item in subjects if item["label"].casefold() == "person"),
                "faces": len(faces),
            },
            "content_bounds": _union_boxes(subjects + faces),
        })
        return result


def create_analysis_provider(config=None):
    if not config:
        return BasicAnalysisProvider()
    provider = config.get("provider", "ultralytics")
    if provider == "basic":
        return BasicAnalysisProvider()
    if provider == "ultralytics":
        return UltralyticsAnalysisProvider(config)
    raise ValueError(f"Unknown analysis provider: {provider}")
