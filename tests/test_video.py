import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image
from unittest.mock import patch

from lora_dataset.captioning import (
    CaptionProvider,
    OpenAICompatibleCaptionProvider,
    apply_video_trigger,
    build_caption_instruction,
    normalize_video_caption_for_profile,
)
from lora_dataset.engine import DatasetEngine
from lora_dataset.profile import DatasetProfileRegistry
from lora_dataset.source import DatasetSource
from lora_dataset.video import (
    VIDEO_EXTENSIONS,
    VideoOrientationExcluded,
    ffmpeg_filter,
    normalize_video_config,
    prepare_video,
    probe_video,
    sample_video_frames,
)
from lora_dataset.transcription import (
    discover_original_video,
    harvester_start_time,
    transcript_for_window,
)


FFMPEG = shutil.which("ffmpeg")


def make_video(path, duration=0.75, size="320x240"):
    if not FFMPEG:
        pytest.skip("FFmpeg is not available")
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=12",
            "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


class FakeVideoCaptionProvider(CaptionProvider):
    def __init__(self):
        self.calls = []

    def caption(self, image_path, instruction, context=None):
        raise AssertionError("Video mode must not call the single-image caption method")

    def caption_video(self, video_path, instruction, context=None):
        self.calls.append((Path(video_path), instruction, dict(context or {})))
        return "a colorful test pattern moves continuously across the frame"


def video_profile():
    return DatasetProfileRegistry().recipe("MiniMax H3", "Concept", "motion_token")


def test_video_source_and_filter_are_deterministic(tmp_path):
    make_video(tmp_path / "b.mp4")
    make_video(tmp_path / "a.mkv")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")
    items = DatasetSource(tmp_path, extensions=VIDEO_EXTENSIONS).discover()
    assert [item.relative_path for item in items] == ["a.mkv", "b.mp4"]
    expression = ffmpeg_filter({
        "fps": 24,
        "width": 1024,
        "height": 768,
        "resize_mode": "crop_to_fill",
        "crop_position": "top",
    })
    assert "fps=24" in expression
    assert "crop=1024:768:(iw-ow)/2:0" in expression

    native = normalize_video_config({
        "resize_mode": "keep_native",
        "width": 64,
        "height": 64,
    })
    assert native["width"] == 0 and native["height"] == 0
    native_expression = ffmpeg_filter(native)
    assert "scale=" not in native_expression
    assert "crop=" not in native_expression
    assert "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=black" in native_expression


def test_keep_native_preserves_source_dimensions(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "prepared.mp4"
    make_video(source)
    metadata = prepare_video(source, output, {
        "duration": 0.5,
        "fps": 6,
        "width": 64,
        "height": 64,
        "resize_mode": "keep_native",
    })
    assert metadata["width"] == 320
    assert metadata["height"] == 240


@pytest.mark.parametrize(
    ("size", "expected_dimensions", "expected_orientation"),
    [
        ("320x240", (96, 64), "landscape"),
        ("240x320", (64, 96), "portrait"),
    ],
)
def test_orientation_normalization_and_exact_frame_count(
    tmp_path, size, expected_dimensions, expected_orientation
):
    source = tmp_path / f"{expected_orientation}-source.mp4"
    output = tmp_path / f"{expected_orientation}-prepared.mp4"
    make_video(source, duration=0.5, size=size)
    config = {
        "fps": 6,
        "target_frame_count": 13,
        "resize_mode": "crop_to_fill",
        "size_strategy": "normalize_by_orientation",
        "landscape_width": 96,
        "landscape_height": 64,
        "portrait_width": 64,
        "portrait_height": 96,
        "orientation_filter": "both",
    }

    metadata = prepare_video(source, output, config)

    assert (metadata["width"], metadata["height"]) == expected_dimensions
    assert metadata["source_orientation"] == expected_orientation
    assert metadata["frames"] == 13
    assert metadata["fps"] == pytest.approx(6.0, abs=0.01)
    assert metadata["duration"] == pytest.approx(13 / 6, abs=0.05)


def test_orientation_filter_rejects_mismatched_clip(tmp_path):
    source = tmp_path / "portrait.mp4"
    make_video(source, size="240x320")
    with pytest.raises(VideoOrientationExcluded, match="source is portrait"):
        prepare_video(source, tmp_path / "prepared.mp4", {
            "orientation_filter": "landscape_only",
            "resize_mode": "crop_to_fill",
            "size_strategy": "normalize_by_orientation",
        })


def test_orientation_normalization_rejects_variable_fit_within():
    with pytest.raises(ValueError, match="preserves variable output dimensions"):
        normalize_video_config({
            "size_strategy": "normalize_by_orientation",
            "resize_mode": "fit_within",
        })


def test_video_engine_prepares_captions_validates_and_resumes(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_video(source / "clip.mov")
    provider = FakeVideoCaptionProvider()
    config = {
        "duration": 0.5,
        "fps": 6,
        "width": 256,
        "height": 256,
        "resize_mode": "crop_to_fill",
        "crop_position": "center",
        "caption_frames": 4,
        "caption_megapixels": 0.1,
        "keep_audio": False,
    }
    first = DatasetEngine(
        source,
        destination,
        video_profile(),
        caption_provider=provider,
        caption_provider_version="fake-video-v1",
        media_type="videos",
        video_config=config,
    ).run("resume")
    output = destination / "dataset" / "clip.mp4"
    caption = destination / "dataset" / "clip.txt"
    assert first["training_ready"] is True
    assert first["media_type"] == "videos"
    assert output.is_file() and caption.is_file()
    assert caption.read_text(encoding="utf-8").startswith("motion_token, ")
    assert provider.calls[0][0] == output
    assert provider.calls[0][2]["media_type"] == "video"
    assert "ordered frames sampled chronologically" in provider.calls[0][1]
    assert 'required trigger is "motion_token"' in provider.calls[0][1]
    metadata = probe_video(output, config)
    assert metadata["width"] == 256 and metadata["height"] == 256
    assert metadata["fps"] == pytest.approx(6.0, abs=0.05)
    frames, _ = sample_video_frames(output, count=4, megapixels=0.1, config=config)
    assert len(frames) == min(4, metadata["frames"])

    second_provider = FakeVideoCaptionProvider()
    second = DatasetEngine(
        source,
        destination,
        video_profile(),
        caption_provider=second_provider,
        caption_provider_version="fake-video-v1",
        media_type="videos",
        video_config=config,
    ).run("resume")
    assert second["processed_this_run"] == 0
    assert second_provider.calls == []


def test_video_instruction_is_temporal_but_image_instruction_is_unchanged():
    recipe = video_profile()
    video_instruction = build_caption_instruction(recipe, "video")
    assert "MiniMax H3 LoRA training" in video_instruction
    assert "distinguish camera movement from subject movement" in video_instruction
    assert "camera remains stationary" in video_instruction
    assert "include all clearly spoken words in quotation marks" in video_instruction
    assert "leaving the shared target style primarily represented by the trigger" in video_instruction
    assert "Use natural English rather than tag-style captions" in video_instruction
    assert "attached image" in build_caption_instruction(recipe)


@pytest.mark.parametrize("model", ["Krea 2", "Anima"])
def test_video_mode_rejects_non_h3_profiles(tmp_path, model):
    source = tmp_path / "source"
    source.mkdir()
    recipe = DatasetProfileRegistry().recipe(model, "Concept", "motion_token")
    engine = DatasetEngine(
        source,
        tmp_path / "project",
        recipe,
        media_type="videos",
        video_config={},
    )
    with pytest.raises(ValueError, match="Select MiniMax H3"):
        engine.run("resume")


def test_h3_profile_rejects_image_media_mode(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    engine = DatasetEngine(source, tmp_path / "project", video_profile())
    with pytest.raises(ValueError, match="Select videos"):
        engine.run("resume")


@pytest.mark.parametrize(
    "bad_caption",
    [
        "the camera appears to move around the subject",
        "the subject probably turns toward the camera",
        "masterpiece, a person walks across a room",
        "a highly detailed professional video of a moving vehicle",
    ],
)
def test_h3_video_caption_validation_rejects_uncertainty_and_quality_tags(bad_caption):
    with pytest.raises(ValueError, match="Video caption contains prohibited"):
        normalize_video_caption_for_profile(bad_caption, video_profile())


def test_video_trigger_is_forced_to_the_exact_caption_prefix():
    recipe = video_profile()
    assert apply_video_trigger("a subject moves, motion_token", recipe) == (
        "motion_token, a subject moves"
    )
    assert apply_video_trigger("motion_token, a subject moves", recipe) == (
        "motion_token, a subject moves"
    )


def test_openai_video_caption_request_contains_ordered_frame_images():
    provider = OpenAICompatibleCaptionProvider({
        "backend": "OpenRouter",
        "api_url": "https://example.invalid/v1",
        "api_key": "test",
        "model_name": "vision-model",
    })
    captured = {}

    def fake_request(messages):
        captured["messages"] = messages
        return "a subject moves from left to right"

    provider._openai_request = fake_request
    frames = [Image.new("RGB", (32, 24), color) for color in ("red", "blue", "green")]
    with patch("lora_dataset.captioning.sample_video_frames", return_value=(frames, {"duration": 5.0})):
        result = provider.caption_video(
            "clip.mp4",
            "Describe the visible motion.",
            {"video_config": {"caption_frames": 3, "caption_megapixels": 0.1}},
        )
    content = captured["messages"][1]["content"]
    assert result == "a subject moves from left to right"
    assert [part["type"] for part in content].count("image_url") == 3
    labels = [
        part["text"] for part in content
        if part["type"] == "text" and part["text"].startswith("Ordered video frame")
    ]
    assert labels == [
        "Ordered video frame 1 of 3:",
        "Ordered video frame 2 of 3:",
        "Ordered video frame 3 of 3:",
    ]


def test_openai_video_caption_request_includes_whisper_evidence():
    provider = OpenAICompatibleCaptionProvider({
        "backend": "OpenRouter",
        "api_url": "https://example.invalid/v1",
        "api_key": "test",
        "model_name": "vision-model",
    })
    captured = {}
    provider._openai_request = lambda messages: captured.setdefault("messages", messages) and "a woman says hello"
    frames = [Image.new("RGB", (32, 24), "black")]
    with patch("lora_dataset.captioning.sample_video_frames", return_value=(frames, {"duration": 5.0})):
        provider.caption_video(
            "clip.mp4",
            "Describe the clip.",
            {"video_config": {}, "audio_evidence": "Hello there."},
        )
    text_parts = [part["text"] for part in captured["messages"][1]["content"] if part["type"] == "text"]
    assert any("Hello there." in text for text in text_parts)


def test_harvester_timestamp_and_transcript_window_alignment(tmp_path):
    clip = tmp_path / "movie__0042_t0000265000.mp4"
    assert harvester_start_time(clip) == 265.0
    transcript = {
        "segments": [
            {"start": 260.0, "end": 264.0, "text": "Too early."},
            {"start": 265.1, "end": 267.0, "text": "Open the door."},
            {"start": 269.5, "end": 271.0, "text": "At the edge."},
            {"start": 272.0, "end": 273.0, "text": "Too late."},
        ]
    }
    assert transcript_for_window(transcript, 265.0, 5.0) == "Open the door. At the edge."


def test_clip_harvester_manifest_discovers_original_video(tmp_path):
    original = tmp_path / "movie.mkv"
    original.write_bytes(b"movie")
    (tmp_path / "clip_harvester_source.json").write_text(
        json.dumps({"schema_version": 1, "source_video": str(original)}), encoding="utf-8"
    )
    assert discover_original_video(tmp_path) == original.resolve()
