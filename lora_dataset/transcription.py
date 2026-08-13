import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path


WHISPER_MODELS = ("tiny.en", "base.en", "small.en", "medium", "large-v3-turbo")
WHISPER_DEVICES = ("auto", "cuda", "cpu")
HARVESTER_SOURCE_MANIFEST = "clip_harvester_source.json"
HARVESTER_TIMESTAMP = re.compile(r"_t(?P<milliseconds>\d{10})(?:_|$)")
TRANSCRIPT_SCHEMA_VERSION = 2
CONTEXT_PADDING_SECONDS = 5.0
MINIMUM_CONTEXT_TARGET_SECONDS = 5.0
MINIMUM_SEGMENT_LOGPROB = -1.0
MAXIMUM_NO_SPEECH_PROBABILITY = 0.6
MAXIMUM_COMPRESSION_RATIO = 2.4
MINIMUM_WORD_PROBABILITY = 0.4
DIALOGUE_MIX_VERSION = "dialogue-mix-v2"


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
        if segment.get("reliable") is False:
            continue
        words = segment.get("words") or []
        if words:
            selected = []
            for word in words:
                word_start = float(word.get("start", 0.0))
                word_end = float(word.get("end", word_start))
                if word_end >= lower and word_start <= upper:
                    selected.append(str(word.get("word") or ""))
            text = "".join(selected).strip()
            if text:
                lines.append(text)
            continue
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        if segment_end < lower or segment_start > upper:
            continue
        text = str(segment.get("text") or "").strip()
        if text:
            lines.append(text)
    return " ".join(lines).strip()


def _serialize_segment(segment, offset=0.0, audio_duration=None):
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    words = [
        {
            "word": str(word.get("word") or ""),
            "start": float(word.get("start", 0.0)) + offset,
            "end": float(word.get("end", word.get("start", 0.0))) + offset,
            "probability": float(word.get("probability", 0.0)),
        }
        for word in segment.get("words") or []
        if str(word.get("word") or "").strip()
    ]
    # Keep cache JSON standards-compliant even if a Whisper-compatible backend
    # omits confidence fields. Missing confidence is deliberately treated as
    # unreliable instead of being accepted on faith.
    average_logprob = float(segment.get("avg_logprob", -99.0))
    no_speech_probability = float(segment.get("no_speech_prob", 1.0))
    compression_ratio = float(segment.get("compression_ratio", 99.0))
    rejection_reasons = []
    if average_logprob < MINIMUM_SEGMENT_LOGPROB:
        rejection_reasons.append("low_segment_probability")
    if no_speech_probability > MAXIMUM_NO_SPEECH_PROBABILITY:
        rejection_reasons.append("probable_non_speech")
    if compression_ratio > MAXIMUM_COMPRESSION_RATIO:
        rejection_reasons.append("repetitive_transcript")
    if words and min(word["probability"] for word in words) < MINIMUM_WORD_PROBABILITY:
        rejection_reasons.append("low_word_probability")
    if audio_duration is not None and end > float(audio_duration) + 0.75:
        rejection_reasons.append("timestamp_outside_audio")
    return {
        "start": start + offset,
        "end": end + offset,
        "text": str(segment.get("text") or "").strip(),
        "avg_logprob": average_logprob,
        "no_speech_prob": no_speech_probability,
        "compression_ratio": compression_ratio,
        "words": words,
        "reliable": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
    }


class WhisperTranscriber:
    def __init__(
        self,
        cache_directory,
        model_name="large-v3-turbo",
        language="en",
        device="auto",
        ffmpeg_path="",
    ):
        if model_name not in WHISPER_MODELS:
            raise ValueError(f"Unknown Whisper model: {model_name}")
        if device not in WHISPER_DEVICES:
            raise ValueError(f"Unknown Whisper device: {device}")
        self.cache_directory = Path(cache_directory).resolve(strict=False)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.language = str(language or "").strip() or None
        self.device = device
        self.ffmpeg_path = str(ffmpeg_path or "").strip()
        self._model = None

    def _source_signature(self, source, window=None):
        source = Path(source).resolve(strict=True)
        stat = source.stat()
        value = (
            f"{TRANSCRIPT_SCHEMA_VERSION}|{source}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"{self.model_name}|{self.language or 'auto'}|{window or 'full'}"
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    def _cache_path(self, source, window=None):
        return self.cache_directory / f"{self._source_signature(source, window)}.json"

    @staticmethod
    def _read_cache(cache_path):
        if cache_path.is_file():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if payload.get("schema_version") == TRANSCRIPT_SCHEMA_VERSION:
                    return payload
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        return None

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
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached
        return self._transcribe_audio(source, cache_path, source, offset=0.0)

    def transcribe_window(self, source, start, duration, padding=CONTEXT_PADDING_SECONDS):
        source = Path(source).resolve(strict=True)
        start = max(0.0, float(start))
        duration = max(0.01, float(duration))
        padding = max(0.0, float(padding))
        window_start = max(0.0, start - padding)
        # Prepared H3 clips are commonly 107 frames / 24 fps (about 4.46s),
        # even when harvested as five-second source chunks. Preserve at least
        # the original five-second target span so Whisper gets a stable,
        # symmetric 15-second context window instead of a decode-sensitive
        # 14.46-second fragment.
        context_target_duration = max(duration, MINIMUM_CONTEXT_TARGET_SECONDS)
        window_duration = context_target_duration + (start - window_start) + padding
        window_key = f"{window_start:.3f}:{window_duration:.3f}:{DIALOGUE_MIX_VERSION}"
        cache_path = self._cache_path(source, window_key)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        with tempfile.TemporaryDirectory(prefix="lora-whisper-window-") as folder:
            audio_path = Path(folder) / "dialogue.wav"
            self._extract_dialogue_window(source, audio_path, window_start, window_duration)
            return self._transcribe_audio(
                audio_path,
                cache_path,
                source,
                offset=window_start,
                context={
                    "window_start": window_start,
                    "window_duration": window_duration,
                    "target_start": start,
                    "target_duration": duration,
                },
            )

    def _transcribe_audio(self, audio_source, cache_path, reported_source, offset=0.0, context=None):
        try:
            import whisper
        except ImportError as error:
            raise RuntimeError(
                "Whisper dialogue captioning requires openai-whisper. Install it in the Python environment that launches ComfyUI."
            ) from error

        model = self._load_model()
        print(f"[LoRA Dataset Whisper] Transcribing {Path(reported_source).name}", flush=True)
        audio = whisper.load_audio(str(audio_source))
        audio_duration = len(audio) / float(whisper.audio.SAMPLE_RATE)
        result = model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            fp16=getattr(model, "device", None) is not None and str(model.device).startswith("cuda"),
            verbose=False,
            condition_on_previous_text=False,
            word_timestamps=True,
            temperature=0.0,
            hallucination_silence_threshold=1.0,
        )
        segments = [
            _serialize_segment(segment, offset=offset, audio_duration=audio_duration)
            for segment in result.get("segments") or []
            if str(segment.get("text") or "").strip()
        ]
        payload = {
            "schema_version": TRANSCRIPT_SCHEMA_VERSION,
            "source": str(Path(reported_source).resolve(strict=False)),
            "model": self.model_name,
            "language": result.get("language") or self.language or "auto",
            "context": dict(context or {}),
            "segments": segments,
        }
        temporary = cache_path.with_name(f".{cache_path.stem}.{uuid.uuid4().hex}.tmp.json")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        accepted = sum(1 for segment in segments if segment["reliable"])
        rejected = len(segments) - accepted
        print(
            f"[LoRA Dataset Whisper] Evidence: {accepted} reliable segment(s), "
            f"{rejected} discarded low-confidence segment(s)",
            flush=True,
        )
        return payload

    def _resolve_ffmpeg(self):
        candidate = self.ffmpeg_path or shutil.which("ffmpeg")
        if not candidate or not Path(candidate).is_file():
            raise RuntimeError("FFmpeg was not found for Whisper context extraction")
        return str(Path(candidate).resolve(strict=False))

    def _resolve_ffprobe(self):
        ffmpeg = Path(self._resolve_ffmpeg())
        sibling = ffmpeg.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        candidate = sibling if sibling.is_file() else shutil.which("ffprobe")
        return str(Path(candidate).resolve(strict=False)) if candidate else ""

    def _audio_channels(self, source):
        ffprobe = self._resolve_ffprobe()
        if not ffprobe:
            return 2
        result = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=channels", "-of", "json", str(source),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            return 2
        try:
            streams = json.loads(result.stdout.decode("utf-8")).get("streams") or []
            return int(streams[0].get("channels") or 2) if streams else 0
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
            return 2

    def _extract_dialogue_window(self, source, destination, start, duration):
        channels = self._audio_channels(source)
        command = [
            self._resolve_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{duration:.6f}",
            "-map", "0:a:0",
        ]
        if channels >= 3:
            # Keep the front stereo information while giving the center dialogue
            # channel full weight. The deliberate 2x aggregate gain proved more
            # intelligible than a normalized mix on quiet theatrical 5.1 tracks.
            command.extend(["-af", "pan=mono|c0=0.5*c0+0.5*c1+1.0*c2"])
        elif channels == 2:
            command.extend(["-af", "pan=mono|c0=0.5*c0+0.5*c1"])
        else:
            command.extend(["-ac", "1"])
        command.extend(["-ar", "16000", str(destination)])
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0 or not destination.is_file():
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Whisper audio context extraction failed: {detail or result.returncode}")

    def close(self):
        self._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
