import base64
import hashlib
import http.client
import io
import json
import re
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from .provider_images import scale_for_provider
from .video import encode_png, sample_video_frames


DEFAULT_PROVIDER_URLS = {
    "Ollama": "http://localhost:11434/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "NanoGPT": "https://nano-gpt.com/api/v1",
    "Kobold": "http://localhost:5001/v1",
}

NANOGPT_VIDEO_BASE64_BUDGET = 3_250_000
NANOGPT_VIDEO_JPEG_QUALITIES = (90, 82, 74, 66, 58, 50)
TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}

SECRET_FIELDS = {"api_key", "openrouter_key", "nanogpt_key"}

MINIMAX_H3_VIDEO_CAPTION_POLICY = """You are a video dataset captioning model for MiniMax H3 LoRA training.

Your job is to create one accurate sidecar caption for each training video. The attached images are ordered frames sampled chronologically from one prepared video. Treat them as temporal evidence for the complete clip, not as separate images.

The caption must describe what is visibly present in the video, how it changes over time, and any dialogue or audible event supplied as transcript evidence.

Write one concise natural-language paragraph. Do not output reasoning, bullets, numbering, metadata, analysis, headings, or explanations.

Describe the main subject or subjects; important visible appearance details needed to distinguish them; the environment and scene; primary subject motion in chronological order; camera movement, if any; important secondary motion such as hair, clothing, vehicles, water, smoke, foliage, or objects moving through the scene; and framing or composition when it materially affects the clip.

For motion-concept training, give particular attention to the demonstrated motion. Describe it clearly and consistently across clips. Use chronological language when useful: describe the initial state, the motion that occurs, and the ending state. Do not narrate every frame.

For a stationary-camera clip, explicitly state that the camera remains stationary when this helps distinguish subject motion from camera motion. For a camera-motion concept, distinguish camera movement from subject movement.

Caption only supported evidence, not assumptions. Do not use uncertain language such as "appears to," "seems to," "probably," or "possibly." Do not add intentions, emotions that are not visibly expressed, production metadata, quality ratings, or speculative context. When reliable transcript evidence is supplied, include all clearly spoken words in quotation marks and place them in the correct point of the chronological action. Mention music, singing, vocalizations, or sound effects only when the transcript evidence explicitly identifies them. Treat automatic transcripts as fallible: omit garbled text and never invent a speaker identity.

Do not use generic quality tags such as "masterpiece," "best quality," "4K," "highly detailed," or "professional." Do not describe a target visual style unless that style is intentionally meant to be caption-conditioned.

For a style LoRA, describe the visible content while leaving the shared target style primarily represented by the trigger. For a motion LoRA, describe enough scene content to separate it from the repeated motion concept, but do not over-caption irrelevant static details.

Use natural English rather than tag-style captions. Keep the caption focused. Accurately identify the subject, scene, motion, camera behavior, and important visible changes that the LoRA should learn around."""


def provider_config_version(config):
    """Return a stable version without putting API secrets in the manifest."""
    safe = {key: value for key, value in config.items() if key not in SECRET_FIELDS}
    canonical = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_caption_instruction(profile, media_type="image"):
    settings = profile.get("settings", {})
    dataset_type = profile.get("dataset_type", "dataset")
    caption_style = settings.get("caption_style", "natural_language")
    output_format = settings.get("caption_output_format", "single_paragraph")
    max_words = int(settings.get("caption_max_words", 120))
    recipe = settings.get("caption_instruction", "Describe the visible media content accurately.")
    extra = settings.get("additional_caption_instructions", "").strip()

    media_type = "video" if str(media_type).casefold() == "video" else "image"
    if media_type == "video":
        trigger = str(profile.get("trigger") or "").strip()
        trigger_instruction = (
            f'The required trigger is "{trigger}". Begin the caption with that exact trigger followed by a comma.'
            if trigger
            else "No trigger word was supplied. Do not invent one."
        )
        parts = [
            MINIMAX_H3_VIDEO_CAPTION_POLICY,
            trigger_instruction,
            f"Dataset type: {dataset_type}. Maximum length: {max_words} words.",
            f"Dataset-type-specific emphasis: {recipe.strip()}",
            "Return only the finished sidecar caption.",
        ]
        if extra:
            parts.append(f"Additional dataset-specific instructions: {extra}")
        return "\n\n".join(parts)

    parts = [
        "Write one direct positive image caption for the attached image.",
        f"Use {caption_style}. Maximum length: {max_words} words.",
        f"Required output format: {output_format}.",
        recipe.strip(),
        "Use concrete declarative visual language. Every sentence must work unchanged as a positive generation prompt.",
        "Return only the caption content in the requested format.",
    ]
    if extra:
        parts.append(f"Additional dataset-specific instructions: {extra}")
    return "\n\n".join(parts)


def strip_reasoning(text):
    value = str(text or "")
    for tag in ("think", "thinking", "reasoning", "reflection", "thought"):
        value = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", value, flags=re.DOTALL | re.IGNORECASE)
    value = re.sub(
        r"(?is)^\s*(thinking|reasoning|reflection|thought process):.*?\n\s*\n",
        "",
        value,
    )
    markers = re.compile(
        r"(?im)^\s*[*_#> ]*(?:final caption|final answer|caption)\s*[:*_]*\s*$\n?"
    )
    matches = list(markers.finditer(value))
    if matches:
        candidate = value[matches[-1].end():].strip()
        if candidate:
            value = candidate
    value = value.strip().strip("`").strip()
    for opening, closing in (("\"", "\""), ("\u201c", "\u201d"), ("'", "'"), ("\u2018", "\u2019")):
        if len(value) >= 2 and value.startswith(opening) and value.endswith(closing):
            return value[1:-1].strip()
    return value


def normalize_caption(text, max_characters=4000):
    caption = strip_reasoning(text)
    lines = [line.strip() for line in caption.splitlines() if line.strip()]
    if any(re.match(r"^(?:[-*\u2022]|\d+[.)])\s+", line) for line in lines):
        raise ValueError("Caption provider returned a list instead of one caption")
    caption = " ".join(lines)
    caption = re.sub(r"\s+", " ", caption).strip()
    if not caption:
        raise ValueError("Caption provider returned an empty caption")
    if len(caption) > int(max_characters):
        raise ValueError(f"Caption exceeds {max_characters} characters")
    if re.match(
        r"(?i)^(?:here(?:'s| is)|i (?:see|can see)|the image (?:shows|depicts)|analysis:|caption:)",
        caption,
    ):
        raise ValueError("Caption provider returned meta commentary")
    if re.search(r"(?i)(?:^|[,.;]\s*)negative prompt\s*:", caption):
        raise ValueError("Caption provider returned a negative prompt")
    return caption


def validate_positive_only_caption(caption):
    # Positive-only applies to the model's visual description, not verbatim
    # dialogue. Spoken lines can legitimately contain words such as "no",
    # "not", or "aren't" and the video policy requires preserving them.
    visual_prose = re.sub(r'"(?:\\.|[^"\\])*"|“[^”]*”', " ", caption)
    negative_assertions = (
        r"\b(?:no|not|never|without|none|nothing|nobody)\b",
        r"\b(?:isn't|aren't|wasn't|weren't|doesn't|don't|didn't|can't|cannot)\b",
        r"\b(?:free|clear)\s+of\b",
        r"\b(?:absent|removed|eliminated|missing|lacks?|lacking)\b",
    )
    if any(
        re.search(pattern, visual_prose, flags=re.IGNORECASE)
        for pattern in negative_assertions
    ):
        raise ValueError("Caption contains negative or absence language instead of present visual content")
    return caption


def validate_direct_caption_language(caption, max_words=120):
    meta_language = (
        r"\bincidental details?\b",
        r"\b(?:training|dataset|caption|prompt)\b",
        r"\b(?:analysis|reasoning|instructions?|explanations?|notes?)\b",
        r"\bidentity\b",
        r"\b(?:remain|stays?|kept)\s+(?:separate|distinct|controllable)\b",
        r"\b(?:separate|distinct)\s+from\b",
        r"\bnot\s+part\s+of\b",
    )
    if any(re.search(pattern, caption, flags=re.IGNORECASE) for pattern in meta_language):
        raise ValueError("Caption contains training rationale or explanatory meta language")
    words = re.findall(r"\b[\w$'-]+\b", caption, flags=re.UNICODE)
    if len(words) > int(max_words):
        raise ValueError(f"Caption exceeds {max_words} words")
    return caption


def normalize_caption_for_profile(text, profile):
    settings = profile.get("settings", {})
    caption = normalize_caption(text, settings.get("caption_max_characters", 4000))
    validation_caption = caption
    trigger = profile.get("trigger", "").strip()
    if trigger:
        validation_caption = re.sub(
            re.escape(trigger), "", validation_caption, count=1, flags=re.IGNORECASE
        ).strip(" ,")
    if settings.get("caption_output_format") != "comma_separated_tags":
        if settings.get("positive_caption_only", True):
            validate_positive_only_caption(validation_caption)
        validate_direct_caption_language(
            validation_caption, settings.get("caption_max_words", 120)
        )
        return caption

    tags = [tag.strip() for tag in caption.split(",") if tag.strip()]
    if len(tags) < 2:
        raise ValueError("Anima caption must contain a comma-separated tag list, not prose")
    if any(len(tag) > 100 for tag in tags):
        raise ValueError("Anima caption contains a prose-length value instead of tags")
    if any(tag.startswith("@") for tag in tags):
        raise ValueError("Anima LoRA recipe does not permit artist tags")
    if settings.get("positive_caption_only", True):
        for tag in tags:
            if trigger and tag.casefold() == trigger.casefold():
                continue
            validate_positive_only_caption(tag.replace("_", " "))
    return ", ".join(tags)


def normalize_video_caption_for_profile(text, profile):
    caption = normalize_caption_for_profile(text, profile)
    prohibited = {
        r"\bappears? to\b": "uncertain language: appears to",
        r"\bseems? to\b": "uncertain language: seems to",
        r"\bprobably\b": "uncertain language: probably",
        r"\bpossibly\b": "uncertain language: possibly",
        r"\bmasterpiece\b": "generic quality tag: masterpiece",
        r"\bbest quality\b": "generic quality tag: best quality",
        r"\b4k\b": "generic quality tag: 4K",
        r"\bhighly detailed\b": "generic quality tag: highly detailed",
        r"\bprofessional\b": "generic quality tag: professional",
    }
    for pattern, reason in prohibited.items():
        if re.search(pattern, caption, flags=re.IGNORECASE):
            raise ValueError(f"Video caption contains prohibited {reason}")
    return caption


def apply_video_trigger(caption, profile):
    trigger = str(profile.get("trigger") or "").strip()
    if not trigger:
        return caption
    cleaned = re.sub(re.escape(trigger), "", caption, count=1, flags=re.IGNORECASE).strip(" ,")
    return f"{trigger}, {cleaned}" if cleaned else trigger


def apply_trigger(caption, profile):
    trigger = profile.get("trigger", "").strip()
    behavior = profile.get("settings", {}).get("trigger_behavior", "prefix")
    if not trigger:
        return caption
    if behavior == "anima_subject_slot":
        tags = [tag.strip() for tag in caption.split(",") if tag.strip()]
        if any(tag.casefold() == trigger.casefold() for tag in tags):
            return ", ".join(tags)
        metadata = re.compile(
            r"(?i)^(?:masterpiece|best quality|good quality|normal quality|low quality|"
            r"worst quality|score_[1-9]|year \d{4}|newest|recent|mid|early|old|highres|"
            r"absurdres|anime screenshot|jpeg artifacts|official art|safe|sensitive|nsfw|explicit)$"
        )
        subject_count = re.compile(r"(?i)^\d+(?:girls?|boys?|others?)$")
        insertion = 0
        while insertion < len(tags) and metadata.match(tags[insertion]):
            insertion += 1
        while insertion < len(tags) and subject_count.match(tags[insertion]):
            insertion += 1
        tags.insert(insertion, trigger)
        return ", ".join(tags)
    if trigger.casefold() in caption.casefold():
        return caption
    separator = ", " if behavior in {"prefix", "contextual"} else " "
    if behavior == "contextual_prefix":
        separator = ", "
    return f"{trigger}{separator}{caption}".strip()


class CaptionProvider(ABC):
    @abstractmethod
    def caption(self, image_path, instruction, context=None):
        raise NotImplementedError

    def caption_video(self, video_path, instruction, context=None):
        raise NotImplementedError("This caption provider does not support ordered video frames")


class OpenAICompatibleCaptionProvider(CaptionProvider):
    def __init__(self, config):
        self.config = dict(config)
        self.backend = self.config.get("backend", "Ollama")
        self.model_name = self.config.get("model_name", "").strip()
        if not self.model_name:
            raise ValueError("A caption model name is required")

    def caption(self, image_path, instruction, context=None):
        image_b64, image_size = self._encode_image(image_path)
        user_content = [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
        system = (
            "You write direct positive image prompts from visible image content. Supplied instructions "
            "are private constraints. Output concrete visual caption content in the requested format."
            "\n\n/no_think"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        try:
            result = self._openai_request(messages)
        except urllib.error.HTTPError as error:
            if error.code == 404 and self.backend == "Ollama":
                result = self._ollama_request(instruction, image_b64)
            else:
                raise RuntimeError(self._http_error_message(error)) from error
        except Exception as error:
            if isinstance(error, RuntimeError):
                raise
            raise RuntimeError(f"Caption request failed for {self.backend}: {error}") from error
        finally:
            self._unload_ollama()
        print(
            f"[LoRA Dataset Captioner] Captioned {Path(image_path).name} with {self.backend} "
            f"at {image_size[0]}x{image_size[1]}",
            flush=True,
        )
        return result

    def caption_video(self, video_path, instruction, context=None):
        context = dict(context or {})
        video_config = context.get("video_config") or {}
        frames, metadata = sample_video_frames(
            video_path,
            count=video_config.get("caption_frames", 8),
            megapixels=video_config.get("caption_megapixels", 0.35),
            config=video_config,
        )
        encoded, frame_encoding = self._encode_video_frames(frames)
        user_content = [{"type": "text", "text": instruction}]
        audio_evidence = str(context.get("audio_evidence") or "").strip()
        if audio_evidence:
            user_content.append({
                "type": "text",
                "text": (
                    "Automatic Whisper transcript aligned to this clip. Use it as audio evidence; "
                    "it may contain recognition errors:\n" + audio_evidence
                ),
            })
        else:
            user_content.append({
                "type": "text",
                "text": "Use the ordered visual frames as the complete evidence for this caption.",
            })
        for index, (media_type, image_b64) in enumerate(encoded, 1):
            user_content.extend([
                {"type": "text", "text": f"Ordered video frame {index} of {len(encoded)}:"},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
            ])
        system = (
            "You write direct positive video-training prompts from ordered visual evidence and aligned "
            "automatic transcript evidence. Infer only motion and temporal progression supported by the frames. Supplied instructions are private "
            "constraints. Return only the requested caption.\n\n/no_think"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        try:
            result = self._openai_request(messages)
        except urllib.error.HTTPError as error:
            if error.code == 404 and self.backend == "Ollama":
                result = self._ollama_request(instruction, encoded)
            else:
                raise RuntimeError(self._http_error_message(error)) from error
        except Exception as error:
            if isinstance(error, RuntimeError):
                raise
            raise RuntimeError(f"Video caption request failed for {self.backend}: {error}") from error
        finally:
            self._unload_ollama()
        print(
            f"[LoRA Dataset Captioner] Captioned {Path(video_path).name} with {self.backend} "
            f"using {len(encoded)} ordered {frame_encoding} frames ({metadata['duration']:.2f}s)",
            flush=True,
        )
        return result

    def _encode_video_frames(self, frames):
        prepared = [scale_for_provider(frame, self.backend).convert("RGB") for frame in frames]
        if self.backend != "NanoGPT":
            return [("image/png", encode_png(frame)) for frame in prepared], "PNG"

        # NanoGPT rejects uploads around 4 MB. Keep the actual base64 image data
        # comfortably below that boundary so the JSON envelope and transcript
        # still have room. JPEG is dramatically smaller than PNG for movie frames.
        working = prepared
        for _resize_attempt in range(10):
            for quality in NANOGPT_VIDEO_JPEG_QUALITIES:
                encoded = [self._encode_jpeg(frame, quality) for frame in working]
                if sum(len(image_b64) for image_b64 in encoded) <= NANOGPT_VIDEO_BASE64_BUDGET:
                    width, height = working[0].size
                    return (
                        [("image/jpeg", image_b64) for image_b64 in encoded],
                        f"JPEG q{quality} {width}x{height}",
                    )
            working = [
                frame.resize(
                    (max(2, int(frame.width * 0.8)), max(2, int(frame.height * 0.8))),
                    Image.Resampling.LANCZOS,
                )
                for frame in working
            ]
        raise RuntimeError(
            "Could not compress NanoGPT video frames below the safe upload limit"
        )

    @staticmethod
    def _encode_jpeg(image, quality):
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True,
            subsampling=2,
        )
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _encode_image(self, image_path):
        with Image.open(image_path) as image:
            prepared = scale_for_provider(image.copy(), self.backend)
            size = prepared.size
            buffer = io.BytesIO()
            prepared.save(buffer, format="PNG", compress_level=6)
        return base64.b64encode(buffer.getvalue()).decode("ascii"), size

    def _active_key(self):
        key = self.config.get("api_key", "").strip()
        if not key and self.backend == "OpenRouter":
            key = self.config.get("openrouter_key", "").strip()
        if not key and self.backend == "NanoGPT":
            key = self.config.get("nanogpt_key", "").strip()
        return key

    def _base_url(self):
        return self.config.get("api_url", "").strip().rstrip("/") or DEFAULT_PROVIDER_URLS[self.backend]

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        active_key = self._active_key()
        if active_key:
            headers["Authorization"] = f"Bearer {active_key}"
            if self.backend == "NanoGPT":
                headers["X-API-Key"] = active_key
        if self.backend == "OpenRouter":
            headers["HTTP-Referer"] = "https://github.com/lokitsar/ComfyUI-Lokitsars-Nodes"
            headers["X-Title"] = "ComfyUI LoRA Dataset Captioner"
        return headers

    def _payload_options(self):
        payload = {
            "model": self.model_name,
            "max_tokens": int(self.config.get("max_tokens", 512)),
            "temperature": 0.2,
            "stop": ["\nAnalysis:", "\nReasoning:", "\nNegative prompt:"],
        }
        seed = int(self.config.get("seed", 0))
        if seed:
            payload["seed"] = seed
        if self.backend == "OpenRouter":
            payload["thinking"] = {"type": "disabled"}
        return payload

    def _openai_request(self, messages):
        payload = self._payload_options()
        payload["messages"] = messages
        request = urllib.request.Request(
            f"{self._base_url()}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        result = self._request_json_with_retries(request)
        return self._message_text(result["choices"][0]["message"])

    def _request_json_with_retries(self, request):
        attempts = max(1, int(self.config.get("request_attempts", 3)))
        timeout = int(self.config.get("timeout", 120))
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code not in TRANSIENT_HTTP_CODES or attempt + 1 >= attempts:
                    raise
                detail = f"HTTP {error.code}"
            except (
                http.client.RemoteDisconnected,
                ConnectionAbortedError,
                ConnectionResetError,
                TimeoutError,
                socket.timeout,
                urllib.error.URLError,
            ) as error:
                if attempt + 1 >= attempts:
                    raise
                detail = str(error)
            delay = min(4, 2 ** attempt)
            print(
                f"[LoRA Dataset Captioner] Transient {self.backend} request failure "
                f"({detail}); retrying in {delay}s ({attempt + 2}/{attempts})",
                flush=True,
            )
            time.sleep(delay)
        raise RuntimeError(f"Caption request failed after {attempts} attempts")

    def _ollama_request(self, instruction, image_b64):
        base = self._base_url()
        if base.endswith("/v1"):
            base = base[:-3]
        options = {
            "temperature": 0.2,
            "num_predict": int(self.config.get("max_tokens", 512)),
        }
        seed = int(self.config.get("seed", 0))
        if seed:
            options["seed"] = seed
        images = image_b64 if isinstance(image_b64, list) else [image_b64]
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": instruction + "\n\n/no_think", "images": images}],
            "stream": False,
            "keep_alive": 0,
            "options": options,
        }
        request = urllib.request.Request(
            f"{base}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=int(self.config.get("timeout", 120))) as response:
            result = json.loads(response.read().decode("utf-8"))
        return self._message_text(result.get("message", {}))

    @staticmethod
    def _message_text(message):
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return (content or message.get("reasoning_content") or message.get("reasoning") or "").strip()

    def _http_error_message(self, error):
        detail = ""
        try:
            payload = json.loads(error.read().decode("utf-8", errors="replace"))
            provider_error = payload.get("error", payload)
            detail = provider_error.get("message", "") if isinstance(provider_error, dict) else str(provider_error)
        except Exception:
            pass
        message = f"Caption request failed: HTTP {error.code} from {self.backend}"
        return f"{message}. {detail}" if detail else message

    def _unload_ollama(self):
        if self.backend != "Ollama":
            return
        base = self._base_url()
        if base.endswith("/v1"):
            base = base[:-3]
        payload = {"model": self.model_name, "keep_alive": 0, "stream": False}
        try:
            request = urllib.request.Request(
                f"{base}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                response.read()
        except Exception as error:
            print(f"[LoRA Dataset Captioner] Could not unload Ollama model: {error}", flush=True)


def create_caption_provider(config):
    return OpenAICompatibleCaptionProvider(config)
