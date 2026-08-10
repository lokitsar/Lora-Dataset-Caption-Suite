import hashlib
import json
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


def crop_config_version(config):
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _parse_ratio(value):
    left, right = str(value).split(":", 1)
    ratio = float(left) / float(right)
    if ratio <= 0:
        raise ValueError(f"Invalid aspect ratio: {value}")
    return ratio


def _expanded_union(detections, width, height, padding_fraction):
    boxes = [item.get("bbox") for item in detections if item.get("bbox")]
    if not boxes:
        return None
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    padding_x = width * padding_fraction
    padding_y = height * padding_fraction
    return (
        max(0.0, left - padding_x),
        max(0.0, top - padding_y),
        min(float(width), right + padding_x),
        min(float(height), bottom + padding_y),
    )


def _position_window(width, height, crop_width, crop_height, protected):
    if protected is None:
        left = (width - crop_width) / 2
        top = (height - crop_height) / 2
        return left, top, left + crop_width, top + crop_height
    p_left, p_top, p_right, p_bottom = protected
    if (p_right - p_left) > crop_width or (p_bottom - p_top) > crop_height:
        return None
    preferred_left = ((p_left + p_right) / 2) - (crop_width / 2)
    preferred_top = ((p_top + p_bottom) / 2) - (crop_height / 2)
    minimum_left = max(0.0, p_right - crop_width)
    maximum_left = min(p_left, width - crop_width)
    minimum_top = max(0.0, p_bottom - crop_height)
    maximum_top = min(p_top, height - crop_height)
    left = min(max(preferred_left, minimum_left), maximum_left)
    top = min(max(preferred_top, minimum_top), maximum_top)
    return left, top, left + crop_width, top + crop_height


class CropProvider(ABC):
    @abstractmethod
    def crop(self, image_path, analysis, profile, context=None):
        raise NotImplementedError


class ProfileSafeCropProvider(CropProvider):
    def __init__(self, config):
        self.config = dict(config)

    def _protected_detections(self, analysis, strategy):
        subjects = list(analysis.get("subjects", []))
        faces = list(analysis.get("faces", []))
        if strategy == "identity_preserving":
            people = [item for item in subjects if item.get("label", "").casefold() == "person"]
            return people + faces
        if strategy == "semantic":
            return subjects + faces
        return []

    def crop(self, image_path, analysis, profile, context=None):
        dimensions = analysis.get("dimensions", {})
        width = int(dimensions.get("width", 0))
        height = int(dimensions.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError("Crop provider received invalid analysis dimensions")

        settings = profile.get("settings", {})
        strategy = settings.get("crop_strategy", "composition_preserving")
        allowed = settings.get("allowed_aspect_ratios", ["1:1"])
        tolerance = float(settings.get("crop_aspect_tolerance", 0.035))
        current_ratio = width / height
        parsed = [(label, _parse_ratio(label)) for label in allowed]
        closest_label, closest_ratio = min(parsed, key=lambda item: abs(item[1] - current_ratio))
        relative_error = abs(closest_ratio - current_ratio) / closest_ratio
        if relative_error <= tolerance:
            return {
                "status": "not_needed_allowed_aspect",
                "strategy": strategy,
                "source_dimensions": [width, height],
                "output_dimensions": [width, height],
                "target_aspect": closest_label,
                "crop_box": [0, 0, width, height],
                "removed_area_fraction": 0.0,
            }

        candidates = []
        protected_items = self._protected_detections(analysis, strategy)
        if strategy in {"identity_preserving", "semantic"} and not protected_items:
            return {
                "status": "skipped_no_protected_subjects",
                "strategy": strategy,
                "source_dimensions": [width, height],
                "output_dimensions": [width, height],
            }
        padding = float(settings.get("crop_subject_padding_fraction", 0.06))
        protected = _expanded_union(protected_items, width, height, padding)
        max_removed = float(settings.get("crop_max_removed_fraction", 0.18))
        minimum_dimension = int(self.config.get("minimum_output_dimension", 512))

        for label, ratio in parsed:
            if current_ratio > ratio:
                crop_height = float(height)
                crop_width = crop_height * ratio
            else:
                crop_width = float(width)
                crop_height = crop_width / ratio
            removed = 1.0 - ((crop_width * crop_height) / (width * height))
            if removed > max_removed:
                continue
            if min(crop_width, crop_height) < minimum_dimension:
                continue
            window = _position_window(width, height, crop_width, crop_height, protected)
            if window is None:
                continue
            candidates.append((removed, label, window))

        if not candidates:
            return {
                "status": "skipped_crop_safety_limits",
                "strategy": strategy,
                "source_dimensions": [width, height],
                "output_dimensions": [width, height],
            }

        removed, target_label, window = min(candidates, key=lambda item: item[0])
        left, top, right, bottom = window
        crop_box = [int(round(left)), int(round(top)), int(round(right)), int(round(bottom))]
        crop_box[0] = max(0, min(crop_box[0], width - 1))
        crop_box[1] = max(0, min(crop_box[1], height - 1))
        crop_box[2] = max(crop_box[0] + 1, min(crop_box[2], width))
        crop_box[3] = max(crop_box[1] + 1, min(crop_box[3], height))
        output_size = [crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]]

        output = Path(image_path)
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        try:
            with Image.open(output) as image:
                cropped = image.crop(tuple(crop_box))
                cropped.save(temporary, format="PNG", compress_level=6)
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()

        return {
            "status": f"cropped_{strategy}",
            "strategy": strategy,
            "source_dimensions": [width, height],
            "output_dimensions": output_size,
            "target_aspect": target_label,
            "crop_box": crop_box,
            "removed_area_fraction": round(removed, 6),
        }


def create_crop_provider(config):
    provider = config.get("provider", "profile_safe_crop")
    if provider == "profile_safe_crop":
        return ProfileSafeCropProvider(config)
    raise ValueError(f"Unknown crop provider: {provider}")
