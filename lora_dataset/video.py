import base64
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from PIL import Image

from .transcription import WHISPER_DEVICES, WHISPER_MODELS


VIDEO_EXTENSIONS = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
CROP_POSITIONS = {
    "center": ("(iw-ow)/2", "(ih-oh)/2"),
    "top": ("(iw-ow)/2", "0"),
    "bottom": ("(iw-ow)/2", "ih-oh"),
    "left": ("0", "(ih-oh)/2"),
    "right": ("iw-ow", "(ih-oh)/2"),
    "top_left": ("0", "0"),
    "top_right": ("iw-ow", "0"),
    "bottom_left": ("0", "ih-oh"),
    "bottom_right": ("iw-ow", "ih-oh"),
}
RESIZE_MODES = ("keep_native", "crop_to_fill", "fit_within", "pad_to_fit", "stretch")
ENCODER_PRESETS = ("fast", "medium", "slow")
SIZE_STRATEGIES = ("single_size", "normalize_by_orientation")
ORIENTATION_FILTERS = ("both", "landscape_only", "portrait_only")


class VideoOrientationExcluded(ValueError):
    """The clip does not match the requested dataset orientation."""


def _even_dimension(value, default):
    return max(2, int(value if value is not None else default) // 2 * 2)


def normalize_video_config(config=None):
    raw = dict(config or {})
    resize_mode = str(raw.get("resize_mode", "fit_within"))
    crop_position = str(raw.get("crop_position", "center"))
    if resize_mode not in RESIZE_MODES:
        raise ValueError(f"Unknown video resize mode: {resize_mode}")
    if crop_position not in CROP_POSITIONS:
        raise ValueError(f"Unknown video crop position: {crop_position}")
    preset = str(raw.get("encoder_preset", "medium"))
    if preset not in ENCODER_PRESETS:
        raise ValueError(f"Unknown FFmpeg encoder preset: {preset}")
    size_strategy = str(raw.get("size_strategy", "single_size"))
    orientation_filter = str(raw.get("orientation_filter", "both"))
    if size_strategy not in SIZE_STRATEGIES:
        raise ValueError(f"Unknown video size strategy: {size_strategy}")
    if orientation_filter not in ORIENTATION_FILTERS:
        raise ValueError(f"Unknown video orientation filter: {orientation_filter}")
    if size_strategy == "normalize_by_orientation" and resize_mode == "fit_within":
        raise ValueError(
            "normalize_by_orientation requires crop_to_fill, pad_to_fit, or stretch; "
            "fit_within preserves variable output dimensions"
        )
    width = _even_dimension(raw.get("width"), 1024)
    height = _even_dimension(raw.get("height"), 1024)
    landscape_width = _even_dimension(raw.get("landscape_width"), 896)
    landscape_height = _even_dimension(raw.get("landscape_height"), 512)
    portrait_width = _even_dimension(raw.get("portrait_width"), 512)
    portrait_height = _even_dimension(raw.get("portrait_height"), 896)
    if resize_mode == "keep_native":
        # Canonicalize ignored controls so changing width/height does not cause a
        # needless manifest rebuild while native sizing is selected.
        width = 0
        height = 0
        landscape_width = 0
        landscape_height = 0
        portrait_width = 0
        portrait_height = 0
    elif size_strategy == "normalize_by_orientation":
        width = 0
        height = 0
    else:
        landscape_width = 0
        landscape_height = 0
        portrait_width = 0
        portrait_height = 0
    target_frame_count = max(0, int(raw.get("target_frame_count", 0)))
    fps = max(1.0, float(raw.get("fps", 24.0)))
    duration = max(0.1, float(raw.get("duration", 5.0)))
    pad_short_video = bool(raw.get("pad_short_video", False))
    if target_frame_count:
        # Duration and padding become derived behavior in exact-frame mode.
        duration = target_frame_count / fps
        pad_short_video = True
    whisper_model = str(raw.get("whisper_model", "large-v3-turbo"))
    whisper_device = str(raw.get("whisper_device", "auto"))
    if whisper_model not in WHISPER_MODELS:
        raise ValueError(f"Unknown Whisper model: {whisper_model}")
    if whisper_device not in WHISPER_DEVICES:
        raise ValueError(f"Unknown Whisper device: {whisper_device}")
    return {
        "ffmpeg_path": str(raw.get("ffmpeg_path", "")).strip(),
        "start_time": max(0.0, float(raw.get("start_time", 0.0))),
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "size_strategy": size_strategy,
        "landscape_width": landscape_width,
        "landscape_height": landscape_height,
        "portrait_width": portrait_width,
        "portrait_height": portrait_height,
        "orientation_filter": orientation_filter,
        "target_frame_count": target_frame_count,
        "resize_mode": resize_mode,
        "crop_position": crop_position,
        "pad_short_video": pad_short_video,
        "keep_audio": bool(raw.get("keep_audio", False)),
        "crf": min(51, max(0, int(raw.get("crf", 18)))),
        "encoder_preset": preset,
        "caption_frames": min(32, max(2, int(raw.get("caption_frames", 8)))),
        "caption_megapixels": min(4.0, max(0.05, float(raw.get("caption_megapixels", 0.35)))),
        "transcribe_audio": bool(raw.get("transcribe_audio", False)),
        "original_video_path": str(raw.get("original_video_path", "")).strip(),
        "whisper_model": whisper_model,
        "whisper_language": str(raw.get("whisper_language", "en")).strip(),
        "whisper_device": whisper_device,
        "schema_version": 5,
    }


def video_config_version(config=None):
    canonical = json.dumps(normalize_video_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def resolve_ffmpeg(config=None):
    configured = normalize_video_config(config).get("ffmpeg_path", "")
    candidate = configured or shutil.which("ffmpeg")
    if not candidate or not Path(candidate).is_file():
        raise RuntimeError("FFmpeg was not found. Add ffmpeg to PATH or set ffmpeg_path in LoRA Dataset Video Prep.")
    return str(Path(candidate).resolve(strict=False))


def resolve_ffprobe(config=None):
    ffmpeg = Path(resolve_ffmpeg(config))
    sibling = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix.casefold() == ".exe" else "ffprobe")
    candidate = sibling if sibling.is_file() else shutil.which("ffprobe")
    if not candidate:
        raise RuntimeError("FFprobe was not found beside FFmpeg or on PATH")
    return str(Path(candidate).resolve(strict=False))


def video_orientation(width, height):
    return "landscape" if int(width) >= int(height) else "portrait"


def target_dimensions(config, source_width=None, source_height=None):
    settings = normalize_video_config(config)
    if settings["resize_mode"] == "keep_native":
        return None
    if settings["size_strategy"] == "normalize_by_orientation":
        if not source_width or not source_height:
            raise ValueError("Source dimensions are required for orientation normalization")
        orientation = video_orientation(source_width, source_height)
        if orientation == "landscape":
            return settings["landscape_width"], settings["landscape_height"]
        return settings["portrait_width"], settings["portrait_height"]
    return settings["width"], settings["height"]


def ffmpeg_filter(config=None, source_width=None, source_height=None):
    settings = normalize_video_config(config)
    dimensions = target_dimensions(settings, source_width, source_height)
    width, height = dimensions or (0, 0)
    filters = [f"fps={settings['fps']:g}"]
    if settings["resize_mode"] == "keep_native":
        # yuv420p requires even dimensions. Padding the right/bottom edge by at
        # most one pixel preserves every source pixel instead of cropping or scaling.
        filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=black")
    elif settings["resize_mode"] == "crop_to_fill":
        crop_x, crop_y = CROP_POSITIONS[settings["crop_position"]]
        filters.extend([
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos",
            f"crop={width}:{height}:{crop_x}:{crop_y}",
        ])
    elif settings["resize_mode"] == "pad_to_fit":
        filters.extend([
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
        ])
    elif settings["resize_mode"] == "fit_within":
        filters.append(
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos"
        )
    else:
        filters.append(f"scale={width}:{height}:flags=lanczos")
    if settings["target_frame_count"] > 0:
        # The output is capped with -frames:v. This pad guarantees that a short
        # source still reaches the exact requested count by holding its last frame.
        pad_duration = settings["target_frame_count"] / settings["fps"]
        filters.append(f"tpad=stop_mode=clone:stop_duration={pad_duration:g}")
    elif settings["pad_short_video"]:
        filters.append(f"tpad=stop_mode=clone:stop_duration={settings['duration']:g}")
    return ",".join(filters)


def _run(command, label):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{label} failed: {detail or f'exit code {result.returncode}'}")
    return result.stdout


def prepare_video(source_path, output_path, config=None):
    settings = normalize_video_config(config)
    source = Path(source_path).resolve(strict=False)
    output = Path(output_path).resolve(strict=False)
    if not source.is_file() or source.suffix.casefold() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported or missing source video: {source}")
    if source == output:
        raise ValueError("Prepared video output cannot overwrite its source")
    source_metadata = probe_video(source, settings)
    orientation = video_orientation(source_metadata["width"], source_metadata["height"])
    orientation_filter = settings["orientation_filter"]
    if orientation_filter != "both" and orientation != orientation_filter.removesuffix("_only"):
        raise VideoOrientationExcluded(
            f"Excluded {source.name}: source is {orientation}, filter is {orientation_filter}"
        )
    dimensions = target_dimensions(
        settings, source_metadata["width"], source_metadata["height"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.tmp.mp4")
    command = [
        resolve_ffmpeg(settings), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{settings['start_time']:.6f}", "-i", str(source), "-map", "0:v:0",
    ]
    if settings["keep_audio"]:
        command.extend(["-map", "0:a?"])
    command.extend([
        "-vf", ffmpeg_filter(settings, source_metadata["width"], source_metadata["height"]),
        "-c:v", "libx264", "-preset", settings["encoder_preset"], "-crf", str(settings["crf"]),
        "-pix_fmt", "yuv420p",
    ])
    if settings["target_frame_count"] > 0:
        exact_duration = settings["target_frame_count"] / settings["fps"]
        command.extend([
            "-frames:v", str(settings["target_frame_count"]),
            "-t", f"{exact_duration:.9f}",
        ])
    else:
        command.extend(["-t", f"{settings['duration']:.6f}"])
    if settings["keep_audio"]:
        # Normalize dataset audio to stereo. Some AAC inputs report six channels
        # without a channel-layout tag; re-encoding those unchanged makes FFmpeg's
        # native AAC encoder fail with "Unsupported channel layout '6 channels'".
        command.extend(["-c:a", "aac", "-ac", "2", "-b:a", "192k"])
    else:
        command.append("-an")
    command.extend(["-movflags", "+faststart", str(temporary)])
    try:
        _run(command, f"FFmpeg preparation for {source.name}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = probe_video(output, settings)
    metadata.update({
        "source_orientation": orientation,
        "target_dimensions": list(dimensions) if dimensions else None,
        "exact_frame_count_requested": settings["target_frame_count"],
    })
    if settings["target_frame_count"] > 0 and metadata["frames"] != settings["target_frame_count"]:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"FFmpeg produced {metadata['frames']} frames; expected exactly "
            f"{settings['target_frame_count']}"
        )
    return metadata


def probe_video(video_path, config=None):
    path = Path(video_path).resolve(strict=False)
    command = [
        resolve_ffprobe(config), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames:stream_tags=rotate:stream_side_data=rotation:format=duration",
        "-of", "json", str(path),
    ]
    payload = json.loads(_run(command, f"FFprobe inspection for {path.name}").decode("utf-8"))
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]
    rotation = int((stream.get("tags") or {}).get("rotate") or 0)
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            rotation = int(side_data.get("rotation") or 0)
            break
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    numerator, _, denominator = str(stream.get("avg_frame_rate", "0/1")).partition("/")
    fps = float(numerator or 0) / max(float(denominator or 1), 1.0)
    return {
        "width": width,
        "height": height,
        "rotation": rotation,
        "fps": round(fps, 6),
        "frames": int(stream.get("nb_frames") or 0),
        "duration": round(float((payload.get("format") or {}).get("duration") or 0.0), 6),
    }


def sample_video_frames(video_path, count=8, megapixels=0.35, config=None):
    path = Path(video_path).resolve(strict=False)
    metadata = probe_video(path, config)
    duration = max(0.001, float(metadata["duration"]))
    count = min(32, max(2, int(count)))
    if metadata["frames"] > 0:
        count = min(count, metadata["frames"])
    maximum_pixels = max(0.05, float(megapixels)) * 1_000_000
    source_pixels = max(1, metadata["width"] * metadata["height"])
    scale = min(1.0, math.sqrt(maximum_pixels / source_pixels))
    width = max(2, int(metadata["width"] * scale) // 2 * 2)
    height = max(2, int(metadata["height"] * scale) // 2 * 2)
    last_frame_time = max(0.0, duration - (1.0 / max(metadata["fps"], 1.0)))
    timestamps = [
        last_frame_time * index / max(1, count - 1)
        for index in range(count)
    ]
    frames = []
    with tempfile.TemporaryDirectory(prefix="lora-video-frames-") as folder:
        for index, timestamp in enumerate(timestamps):
            frame_path = Path(folder) / f"frame-{index:03d}.png"
            command = [
                resolve_ffmpeg(config), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{timestamp:.6f}", "-i", str(path), "-frames:v", "1",
                "-vf", f"scale={width}:{height}:flags=lanczos", str(frame_path),
            ]
            _run(command, f"Frame sampling for {path.name}")
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB").copy())
    if not frames:
        raise RuntimeError(f"Could not sample caption frames from {path}")
    return frames, metadata


def encode_png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
