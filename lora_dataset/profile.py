import copy
import hashlib
import json
from pathlib import Path


def _merge(base, overrides):
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class DatasetProfileRegistry:
    def __init__(self, profile_path=None):
        path = Path(profile_path) if profile_path else Path(__file__).with_name("profiles.json")
        with path.open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)

    @property
    def models(self):
        return tuple(self.config["models"])

    @property
    def dataset_types(self):
        return ("character", "style", "concept")

    def display_name(self, model):
        return self.config["models"][model]["display_name"]

    def model_for_display_name(self, display_name):
        for key, value in self.config["models"].items():
            if value["display_name"] == display_name:
                return key
        if display_name in self.config["models"]:
            return display_name
        raise ValueError(f"Unknown training model: {display_name}")

    def recipe(self, model, dataset_type, trigger, overrides=None):
        model_key = self.model_for_display_name(model)
        dataset_key = str(dataset_type).strip().lower()
        model_config = self.config["models"][model_key]
        if dataset_key not in model_config["datasets"]:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        settings = _merge(model_config.get("defaults", {}), model_config["datasets"][dataset_key])
        if overrides:
            settings = _merge(settings, overrides)
        payload = {
            "model": model_key,
            "model_display_name": model_config["display_name"],
            "dataset_type": dataset_key,
            "trigger": str(trigger).strip(),
            "settings": settings,
            "schema_version": self.config["schema_version"],
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload["profile_version"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        payload["profile_id"] = f"{model_key}:{dataset_key}"
        return payload
