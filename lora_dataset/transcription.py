import gc
import hashlib
import json
import os
import re
import uuid
from pathlib import Path


WHISPER_MODELS = ("tiny.en", "base.en", "small.en", "medium", "large-v3-turbo")
WHISPER_DEVICES = ("auto", "cuda", "cpu")
HARVESTER_SOURCE_MANIFEST = "clip_harvester_source.json"
HARVESTER_TIMESTAMP = re.compile(r"_t(?P<milliseconds>\d{10})(?:_|$)")


def harvester_start_time(path):
    match = HARVESTER_TIMESTAMP.search(Path(path).stem)
    return int(match.group("milliseconds")) / 1000.0 if match else None


def discover_original_video(source_directory, configured_path=""):
    configured = str(configured_path or "").strip().strip('"')
    if configured:
        path = Path(os.path.expandvars(os.path.expanduser(configured))).resolve(strict=False)
        if not path.is_file():
            raise ValueError(f"Whisper original video was not found: {path}")
        return path

    manifest = Path(source_directory).resolve(strict=False) / HARVESTER_SOURCE_MANIFEST
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read Clip Harvester source manifest: {manifest}") from error
    source = Path(str(payload.get("source_video") or "")).resolve(strict=False)
    return source if source.is_file() else None


def transcript_for_window(transcript, start, duration, margin=0.2):
    lower = max(0.0, float(start) - float(margin))
    upper = float(start) + max(0.0, float(duration)) + float(margin)
    lines = []
    for segment in transcript.get("segments") or []:
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        if segment_end < lower or segment_start > upper:
            continue
        text = str(segment.get("text") or "").strip()
        if text:
            lines.append(text)
    return " ".join(lines).strip()


class WhisperTranscriber:
    def __init__(self, cache_directory, model_name="small.en", language="en", device="auto"):
        if model_name not in WHISPER_MODELS:
            raise ValueError(f"Unknown Whisper model: {model_name}")
        if device not in WHISPER_DEVICES:
            raise ValueError(f"Unknown Whisper device: {device}")
        self.cache_directory = Path(cache_directory).resolve(strict=False)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.language = str(language or "").strip() or None
        self.device = device
        self._model = None

    def _source_signature(self, source):
        source = Path(source).resolve(strict=True)
        stat = source.stat()
        value = f"{source}|{stat.st_size}|{stat.st_mtime_ns}|{self.model_name}|{self.language or 'auto'}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    def _cache_path(self, source):
        return self.cache_directory / f"{self._source_signature(source)}.json"

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            import whisper
        except ImportError as error:
            raise RuntimeError(
                "Whisper dialogue captioning requires openai-whisper. Install it in the Python environment that launches ComfyUI."
            ) from error
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[LoRA Dataset Whisper] Loading {self.model_name} on {device}", flush=True)
        self._model = whisper.load_model(self.model_name, device=device)
        return self._model

    def transcribe(self, source):
        source = Path(source).resolve(strict=True)
        cache_path = self._cache_path(source)
        if cache_path.is_file():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if payload.get("schema_version") == 1:
                    return payload
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass

        model = self._load_model()
        print(f"[LoRA Dataset Whisper] Transcribing {source.name}", flush=True)
        result = model.transcribe(
            str(source),
            language=self.language,
            task="transcribe",
            fp16=getattr(model, "device", None) is not None and str(model.device).startswith("cuda"),
            verbose=False,
            condition_on_previous_text=True,
        )
        payload = {
            "schema_version": 1,
            "source": str(source),
            "model": self.model_name,
            "language": result.get("language") or self.language or "auto",
            "segments": [
                {
                    "start": float(segment.get("start", 0.0)),
                    "end": float(segment.get("end", 0.0)),
                    "text": str(segment.get("text") or "").strip(),
                }
                for segment in result.get("segments") or []
                if str(segment.get("text") or "").strip()
            ],
        }
        temporary = cache_path.with_name(f".{cache_path.stem}.{uuid.uuid4().hex}.tmp.json")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        return payload

    def close(self):
        self._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
