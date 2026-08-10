import hashlib
import json
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from .analysis import _normalize_box, _resolve_ultralytics_model


VERIFICATION_SCHEMA_VERSION = 2
DEFAULT_MINIMUM_LUMINANCE_CORRELATION = 0.90


def cleanup_verifier_config_version(config):
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def prepare_fidelity_reference(image_path, megapixels=1.0):
    with Image.open(image_path) as image:
        reference = image.convert("RGB")
        limit = max(0.1, float(megapixels)) * 1_000_000
        if reference.width * reference.height > limit:
            scale = (limit / (reference.width * reference.height)) ** 0.5
            size = (
                max(1, int(round(reference.width * scale))),
                max(1, int(round(reference.height * scale))),
            )
            reference = reference.resize(size, Image.Resampling.LANCZOS)
        return reference.copy()


def _structural_similarity(before_gray, after_gray):
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(before_gray, after_gray, data_range=1.0))
    except ImportError:
        before_centered = before_gray - float(before_gray.mean())
        after_centered = after_gray - float(after_gray.mean())
        denominator = float(
            np.sqrt(np.sum(before_centered ** 2) * np.sum(after_centered ** 2))
        )
        if denominator == 0:
            return 1.0 if np.allclose(before_gray, after_gray) else 0.0
        return float(np.sum(before_centered * after_centered) / denominator)


def _luminance_correlation(before_gray, after_gray):
    before_centered = before_gray - float(before_gray.mean())
    after_centered = after_gray - float(after_gray.mean())
    denominator = float(
        np.sqrt(np.sum(before_centered ** 2) * np.sum(after_centered ** 2))
    )
    if denominator == 0:
        return 1.0 if np.allclose(before_gray, after_gray) else 0.0
    return float(np.sum(before_centered * after_centered) / denominator)


def fidelity_metrics(reference_image, cleaned_image, pixel_change_threshold=0.30):
    reference_aspect = reference_image.width / reference_image.height
    cleaned_aspect = cleaned_image.width / cleaned_image.height
    resized = reference_image.resize(cleaned_image.size, Image.Resampling.LANCZOS)
    before = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0
    after = np.asarray(cleaned_image.convert("RGB"), dtype=np.float32) / 255.0
    delta = np.abs(before - after)
    before_gray = np.dot(before, [0.2126, 0.7152, 0.0722])
    after_gray = np.dot(after, [0.2126, 0.7152, 0.0722])
    return {
        "structural_similarity": round(_structural_similarity(before_gray, after_gray), 6),
        "luminance_correlation": round(
            _luminance_correlation(before_gray, after_gray), 6
        ),
        "mean_absolute_difference": round(float(delta.mean()), 6),
        "changed_area_fraction": round(
            float(np.mean(np.max(delta, axis=2) > float(pixel_change_threshold))), 6
        ),
        "aspect_ratio_delta": round(
            abs(reference_aspect - cleaned_aspect) / reference_aspect, 6
        ),
        "reference_dimensions": [reference_image.width, reference_image.height],
        "cleaned_dimensions": [cleaned_image.width, cleaned_image.height],
    }


class CleanupVerifier(ABC):
    @abstractmethod
    def verify(self, reference_image, cleaned_image_path, context=None):
        raise NotImplementedError


class UltralyticsCleanupVerifier(CleanupVerifier):
    def __init__(self, config):
        self.config = dict(config)
        self.model = None

    def _load_model(self):
        configured = str(self.config.get("watermark_model", "none")).strip()
        if not configured or configured.casefold() == "none":
            return None
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(str(_resolve_ultralytics_model(configured)))
        return self.model

    def _detect(self, image_path, image):
        model = self._load_model()
        if model is None:
            return []
        results = model.predict(
            source=str(image_path),
            conf=float(self.config.get("confidence", 0.2)),
            imgsz=int(self.config.get("image_size", 640)),
            device=self.config.get("device", "cpu"),
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []
        detections = []
        boxes = results[0].boxes
        names = results[0].names
        for index in range(len(boxes)):
            label = str(names.get(int(boxes.cls[index].item()), "watermark"))
            pixels, normalized = _normalize_box(
                boxes.xyxy[index].tolist(), image.width, image.height
            )
            detections.append({
                "label": label,
                "confidence": round(float(boxes.conf[index].item()), 6),
                "bbox": pixels,
                "bbox_normalized": normalized,
            })
        return detections

    def verify(self, reference_image, cleaned_image_path, context=None):
        with Image.open(cleaned_image_path) as image:
            cleaned = image.convert("RGB")
            cleaned.load()
        metrics = fidelity_metrics(
            reference_image,
            cleaned,
            self.config.get("pixel_change_threshold", 0.30),
        )
        limits = {
            "minimum_structural_similarity": float(
                self.config.get("minimum_structural_similarity", 0.72)
            ),
            "maximum_mean_absolute_difference": float(
                self.config.get("maximum_mean_absolute_difference", 0.12)
            ),
            "maximum_changed_area_fraction": float(
                self.config.get("maximum_changed_area_fraction", 0.20)
            ),
            "maximum_aspect_ratio_delta": float(
                self.config.get("maximum_aspect_ratio_delta", 0.02)
            ),
            "minimum_luminance_correlation": float(
                self.config.get(
                    "minimum_luminance_correlation",
                    DEFAULT_MINIMUM_LUMINANCE_CORRELATION,
                )
            ),
        }
        fidelity_failures = []
        fidelity_observations = []
        structure_correlated = (
            metrics["luminance_correlation"] >= limits["minimum_luminance_correlation"]
        )
        if metrics["structural_similarity"] < limits["minimum_structural_similarity"]:
            if structure_correlated:
                fidelity_observations.append(
                    "low_structural_similarity_tolerated_by_luminance_correlation"
                )
            else:
                fidelity_failures.append("structural_similarity_below_limit")
        if metrics["mean_absolute_difference"] > limits["maximum_mean_absolute_difference"]:
            if structure_correlated:
                fidelity_observations.append(
                    "mean_absolute_difference_tolerated_as_global_tonal_change"
                )
            else:
                fidelity_failures.append("mean_absolute_difference_above_limit")
        if metrics["changed_area_fraction"] > limits["maximum_changed_area_fraction"]:
            fidelity_failures.append("changed_area_fraction_above_limit")
        if metrics["aspect_ratio_delta"] > limits["maximum_aspect_ratio_delta"]:
            fidelity_failures.append("aspect_ratio_changed")

        detections = self._detect(cleaned_image_path, cleaned)
        residual = bool(detections)
        excessive_change = bool(fidelity_failures)
        if residual and excessive_change:
            status = "residual_artifact_and_excessive_change"
        elif residual:
            status = "residual_artifact"
        elif excessive_change:
            status = "excessive_change"
        else:
            status = "verified_clean"
        return {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "status": status,
            "passed": status == "verified_clean",
            "residual_detections": detections,
            "residual_detection_count": len(detections),
            "fidelity_metrics": metrics,
            "fidelity_limits": limits,
            "fidelity_failures": fidelity_failures,
            "fidelity_observations": fidelity_observations,
        }


def create_cleanup_verifier(config):
    provider = config.get("provider", "ultralytics_cleanup_verifier")
    if provider == "ultralytics_cleanup_verifier":
        return UltralyticsCleanupVerifier(config)
    raise ValueError(f"Unknown cleanup verifier: {provider}")
