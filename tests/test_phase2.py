import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from lora_dataset.captioning import (
    CaptionProvider,
    OpenAICompatibleCaptionProvider,
    apply_trigger,
    build_caption_instruction,
    normalize_caption,
    normalize_caption_for_profile,
    provider_config_version,
)
from lora_dataset.cleanup import CleanupProvider, DEFAULT_KLEIN_CLEANUP_PROMPT
from lora_dataset.engine import DatasetEngine
from lora_dataset.manifest import DatasetManifest
from lora_dataset.nodes import DatasetBuilderNode, DatasetKleinCleanupProviderNode
from lora_dataset.profile import DatasetProfileRegistry


def make_image(path, size=(48, 32)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (30, 60, 90)).save(path)


def profile(dataset_type="Character", extra=""):
    overrides = {"additional_caption_instructions": extra} if extra else None
    return DatasetProfileRegistry().recipe("Krea 2", dataset_type, "test_subject", overrides)


class FakeCaptionProvider(CaptionProvider):
    def __init__(self, response="a person standing beside a blue vehicle"):
        self.response = response
        self.calls = []

    def caption(self, image_path, instruction, context=None):
        with Image.open(image_path) as image:
            self.calls.append({
                "path": Path(image_path),
                    "format": image.format,
                    "pixel": image.convert("RGB").getpixel((0, 0)),
                "instruction": instruction,
                "context": context,
            })
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeCleanupProvider(CleanupProvider):
    def __init__(self):
        self.calls = []

    def clean(self, image_path, context=None):
        path = Path(image_path)
        with Image.open(path) as image:
            size = image.size
        Image.new("RGB", size, (1, 2, 3)).save(path, format="PNG")
        self.calls.append({"path": path, "context": context})
        return {"status": "cleaned_universal"}


class SelectiveFailureCaptionProvider(CaptionProvider):
    def caption(self, image_path, instruction, context=None):
        if Path(image_path).stem == "bad":
            raise RuntimeError("temporary provider failure")
        return "first visible scene"


def test_provider_generates_caption_from_normalized_png_with_profile_recipe(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "camera.jpeg")
    provider = FakeCaptionProvider()
    recipe = profile(extra="Mention the visible handheld tool if present.")

    result = DatasetEngine(
        source,
        destination,
        recipe,
        caption_provider=provider,
        caption_provider_version="fake-v1",
    ).run("resume")

    assert result["complete"] == 1
    assert provider.calls[0]["format"] == "PNG"
    assert provider.calls[0]["path"].suffix == ".png"
    assert "Maximum length: 110 words" in provider.calls[0]["instruction"]
    assert "Focus on visible factors that vary" in provider.calls[0]["instruction"]
    assert "Do not routinely repeat invariant facial features" in provider.calls[0]["instruction"]
    assert "Mention the visible handheld tool if present." in provider.calls[0]["instruction"]
    assert "Use concrete declarative visual language" in provider.calls[0]["instruction"]
    assert "Return only the caption content" in provider.calls[0]["instruction"]
    assert "watermark" not in provider.calls[0]["instruction"].casefold()
    assert "overlay" not in provider.calls[0]["instruction"].casefold()
    assert "incidental details" not in provider.calls[0]["instruction"].casefold()
    assert "identity" not in provider.calls[0]["instruction"].casefold()
    caption = (destination / "dataset" / "camera.txt").read_text(encoding="utf-8")
    assert caption == "test_subject, a person standing beside a blue vehicle"


def test_builder_node_runs_the_pending_dataset_automatically():
    required = DatasetBuilderNode.INPUT_TYPES()["required"]
    assert "max_items" not in required
    assert "cleanup_provider" in DatasetBuilderNode.INPUT_TYPES()["optional"]


def test_klein_node_uses_comfy_registered_model_dropdowns():
    registered = {
        "diffusion_models": ["other.safetensors", "flux-2-klein-9b.safetensors"],
        "text_encoders": ["qwen_3_8b_fp8mixed.safetensors", "other-clip.safetensors"],
        "vae": ["other-vae.safetensors", "flux2-vae.safetensors"],
    }
    fake_folder_paths = SimpleNamespace(
        get_filename_list=lambda folder_name: registered[folder_name]
    )
    with patch.dict(sys.modules, {"folder_paths": fake_folder_paths}):
        required = DatasetKleinCleanupProviderNode.INPUT_TYPES()["required"]

    assert required["diffusion_model"][0] == [
        "flux-2-klein-9b.safetensors",
        "other.safetensors",
    ]
    assert required["text_encoder"][0][0] == "qwen_3_8b_fp8mixed.safetensors"
    assert required["vae"][0][0] == "flux2-vae.safetensors"
    assert "other-vae.safetensors" in required["vae"][0]


def test_cleanup_runs_before_captioning_and_unlocks_training_ready(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "marked.jpg")
    cleaner = FakeCleanupProvider()
    captioner = FakeCaptionProvider("a clean visible scene")

    result = DatasetEngine(
        source,
        destination,
        profile(),
        caption_provider=captioner,
        caption_provider_version="caption-v1",
        cleanup_provider=cleaner,
        cleanup_provider_version="cleanup-v1",
    ).run("resume")

    assert len(cleaner.calls) == 1
    assert captioner.calls[0]["pixel"] == (1, 2, 3)
    assert result["watermark_audit_complete"]
    assert result["training_ready"]
    record = DatasetManifest(result["manifest"]).records()[0]
    assert record["watermark_status"] == "cleaned_universal"
    assert record["cleanup_provider_version"] == "cleanup-v1"


def test_universal_klein_prompt_is_preservation_first():
    assert "Remove every visible watermark" in DEFAULT_KLEIN_CLEANUP_PROMPT
    assert "Preserve everything else exactly" in DEFAULT_KLEIN_CLEANUP_PROMPT
    assert "If no removable overlay is present" in DEFAULT_KLEIN_CLEANUP_PROMPT


def test_resume_adds_new_images_without_recaptioning_complete_items(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "one.png")
    DatasetEngine(
        source,
        destination,
        profile(),
        caption_provider=FakeCaptionProvider("first visible scene"),
        caption_provider_version="provider-a",
    ).run("resume")

    same = FakeCaptionProvider("should not be used")
    unchanged = DatasetEngine(
        source,
        destination,
        profile(),
        caption_provider=same,
        caption_provider_version="provider-a",
    ).run("resume")
    assert unchanged["processed_this_run"] == 0
    assert not same.calls

    make_image(source / "two.png", size=(52, 36))
    changed = FakeCaptionProvider("second visible scene")
    added = DatasetEngine(
        source,
        destination,
        profile(),
        caption_provider=changed,
        caption_provider_version="provider-b",
    ).run("resume")
    assert added["processed_this_run"] == 1
    assert len(changed.calls) == 1
    assert (destination / "dataset" / "one.txt").read_text(encoding="utf-8") == (
        "test_subject, first visible scene"
    )
    assert (destination / "dataset" / "two.txt").read_text(encoding="utf-8") == (
        "test_subject, second visible scene"
    )

    rebuilt = DatasetEngine(
        source,
        destination,
        profile(),
        caption_provider=FakeCaptionProvider("rewritten visible scene"),
        caption_provider_version="provider-c",
        force_rebuild_revision=1,
    ).run("force_rebuild")
    assert rebuilt["processed_this_run"] == 2
    assert (destination / "dataset" / "one.txt").read_text(encoding="utf-8") == (
        "test_subject, rewritten visible scene"
    )
    assert (destination / "dataset" / "two.txt").read_text(encoding="utf-8") == (
        "test_subject, rewritten visible scene"
    )


def test_failed_caption_can_be_retried_without_duplicate_outputs(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "one.webp")
    failed = DatasetEngine(
        source,
        destination,
        profile(),
        caption_provider=FakeCaptionProvider(RuntimeError("temporary provider failure")),
        caption_provider_version="provider-a",
    ).run("resume")
    assert failed["failed"] == 1

    recovered = DatasetEngine(
        source,
        destination,
        profile(),
        caption_provider=FakeCaptionProvider("a centered portrait"),
        caption_provider_version="provider-a",
    ).run("reprocess_failed")
    assert recovered["complete"] == 1
    assert recovered["failed"] == 0
    assert list((destination / "dataset").glob("*.png")) == [destination / "dataset" / "one.png"]
    assert list((destination / "dataset").glob("*.txt")) == [destination / "dataset" / "one.txt"]


def test_reprocess_failed_preserves_complete_caption_and_also_adds_new_image(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "bad.png", size=(48, 32))
    make_image(source / "good.png", size=(52, 36))
    first = DatasetEngine(
        source,
        destination,
        profile(),
        caption_provider=SelectiveFailureCaptionProvider(),
        caption_provider_version="provider-a",
    ).run("resume")
    assert first["complete"] == 1
    assert first["failed"] == 1
    original_good_caption = (destination / "dataset" / "good.txt").read_bytes()

    make_image(source / "new.png", size=(56, 40))
    replacement = FakeCaptionProvider("second visible scene")
    retried = DatasetEngine(
        source,
        destination,
        profile(),
        caption_provider=replacement,
        caption_provider_version="provider-b",
    ).run("reprocess_failed")

    assert retried["processed_this_run"] == 2
    assert retried["complete"] == 3
    assert retried["failed"] == 0
    assert len(replacement.calls) == 2
    assert (destination / "dataset" / "good.txt").read_bytes() == original_good_caption
    assert (destination / "dataset" / "bad.txt").read_text(encoding="utf-8") == (
        "test_subject, second visible scene"
    )
    assert (destination / "dataset" / "new.txt").read_text(encoding="utf-8") == (
        "test_subject, second visible scene"
    )


def test_caption_normalization_strips_reasoning_and_rejects_bad_outputs():
    assert normalize_caption("<think>hidden</think>\nA visible red bicycle.") == "A visible red bicycle."
    assert normalize_caption("Final caption:\nA dog running on grass.") == "A dog running on grass."
    assert normalize_caption('"A dog running on grass."') == "A dog running on grass."
    assert normalize_caption('A man says, "No."') == 'A man says, "No."'

    for invalid in (
        "",
        "- first item",
        "Here is the caption: a dog",
        "Caption: a dog",
        "A dog. Negative prompt: blurry",
    ):
        try:
            normalize_caption(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid caption to fail: {invalid!r}")


def test_positive_caption_validation_rejects_negative_and_absence_language():
    recipe = profile()
    invalid = (
        "A woman crouches on a beach with no visible overlay text.",
        "A woman crouches on a beach without any watermark.",
        "A woman crouches on a beach; a logo is not visible.",
        "A woman crouches on a beach that is free of text.",
        "A beach scene; any text or overlay artifacts are not part of the scene.",
    )
    for caption in invalid:
        try:
            normalize_caption_for_profile(caption, recipe)
        except ValueError as error:
            assert "negative or absence language" in str(error)
        else:
            raise AssertionError(f"Expected absence language to fail: {caption!r}")
    assert normalize_caption_for_profile(
        "A woman crouches on a pebbled beach beneath a clear blue sky.", recipe
    ).endswith("clear blue sky.")


def test_caption_validation_rejects_training_rationale_and_excess_detail():
    recipe = profile()
    meta = (
        "Incidental details such as the choker and bedding remain separate from the person's identity."
    )
    try:
        normalize_caption_for_profile(meta, recipe)
    except ValueError as error:
        assert "training rationale" in str(error)
    else:
        raise AssertionError("Expected explanatory training rationale to fail")

    overlong = " ".join(["visible"] * 111)
    try:
        normalize_caption_for_profile(overlong, recipe)
    except ValueError as error:
        assert "exceeds 110 words" in str(error)
    else:
        raise AssertionError("Expected over-detailed caption to fail")


def test_invalid_negative_caption_is_retried_once_with_fresh_instruction(tmp_path):
    class SequenceCaptionProvider(CaptionProvider):
        def __init__(self):
            self.responses = [
                "A woman crouches on a beach with no visible overlay text.",
                "A woman crouches on a pebbled beach beneath a clear blue sky.",
            ]
            self.calls = []

        def caption(self, image_path, instruction, context=None):
            self.calls.append({"instruction": instruction, "context": context})
            return self.responses.pop(0)

    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "sample.png")
    provider = SequenceCaptionProvider()
    result = DatasetEngine(
        source,
        destination,
        profile(),
        caption_provider=provider,
        caption_provider_version="sequence-v1",
    ).run("resume")

    assert result["complete"] == 1
    assert len(provider.calls) == 2
    assert provider.calls[1]["context"]["validation_retry"] is True
    assert provider.calls[1]["context"]["validation_retry_attempt"] == 1
    assert "negative or absence language" not in provider.calls[1]["instruction"]
    assert "affirmative language" in provider.calls[1]["instruction"]
    record = DatasetManifest(result["manifest"]).records()[0]
    assert record["caption_status"] == "generated_after_validation_retry"
    assert (destination / "dataset" / "sample.txt").read_text(encoding="utf-8") == (
        "test_subject, A woman crouches on a pebbled beach beneath a clear blue sky."
    )


def test_dataset_validator_flags_negative_language_in_existing_sidecar(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "sample.png")
    (source / "sample.txt").write_text(
        "A woman on a beach with no visible overlay text.", encoding="utf-8"
    )
    result = DatasetEngine(source, destination, profile()).run("resume")
    assert result["failed"] == 1
    record = DatasetManifest(result["manifest"]).records()[0]
    assert "caption_negative_or_absence_language" in record["error"]


def test_profile_prompts_are_specific_and_trigger_is_applied_locally():
    character = build_caption_instruction(profile("Character"))
    style = build_caption_instruction(profile("Style"))
    concept = build_caption_instruction(profile("Concept"))
    assert character != style != concept
    assert "Focus on visible factors that vary" in character
    assert "Do not routinely repeat invariant facial features" in character
    assert "composition, spatial layout" in style
    assert "Make the depicted interaction unambiguous" in concept
    assert apply_trigger("a visible subject", profile()) == "test_subject, a visible subject"
    assert apply_trigger("test_subject, a visible subject", profile()) == "test_subject, a visible subject"


def test_anima_recipes_use_native_tag_format_and_trigger_slot():
    registry = DatasetProfileRegistry()
    character = registry.recipe("Anima", "Character", "hero_token")
    style = registry.recipe("Anima", "Style", "style_token")
    concept = registry.recipe("Anima", "Concept", "concept_token")

    prompts = [build_caption_instruction(recipe) for recipe in (character, style, concept)]
    assert len(set(prompts)) == 3
    assert "comma_separated_tags" in prompts[0]
    assert "1girl or 1boy" in prompts[0]
    assert "composition, viewpoint" in prompts[1]
    assert "relationships needed to describe the depicted interaction" in prompts[2]

    caption = "best quality, year 2025, 1girl, solo, red dress, smile"
    assert apply_trigger(caption, character) == (
        "best quality, year 2025, 1girl, hero_token, solo, red dress, smile"
    )
    assert apply_trigger("1girl, hero_token, smile", character) == "1girl, hero_token, smile"
    assert normalize_caption_for_profile("1girl, solo, red dress, smile", character) == (
        "1girl, solo, red dress, smile"
    )
    for invalid in (
        "A girl wearing a red dress.",
        "1girl, @known artist, smile",
        "best quality, no_humans, beach",
    ):
        try:
            normalize_caption_for_profile(invalid, character)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid Anima caption to fail: {invalid!r}")


def test_provider_config_version_excludes_api_secrets():
    base = {
        "backend": "NanoGPT",
        "api_url": "https://nano-gpt.com/api/v1",
        "api_key": "secret-one",
        "nanogpt_key": "saved-one",
        "openrouter_key": "unused-one",
        "model_name": "vision-model",
        "max_tokens": 512,
    }
    changed_secrets = dict(base, api_key="secret-two", nanogpt_key="saved-two", openrouter_key="unused-two")
    assert provider_config_version(base) == provider_config_version(changed_secrets)
    assert provider_config_version(base) != provider_config_version(dict(base, model_name="other-model"))


def test_nanogpt_caption_provider_scales_payload_and_sends_both_auth_headers(tmp_path):
    image_path = tmp_path / "large.png"
    make_image(image_path, size=(2000, 1000))
    provider = OpenAICompatibleCaptionProvider({
        "backend": "NanoGPT",
        "api_url": "https://nano-gpt.com/api/v1",
        "api_key": "test-secret",
        "model_name": "vision-model",
    })
    encoded, size = provider._encode_image(image_path)
    headers = provider._headers()
    assert encoded
    assert size[0] * size[1] <= 1_000_000
    assert headers["Authorization"] == "Bearer test-secret"
    assert headers["X-API-Key"] == "test-secret"


def test_existing_manifest_is_migrated_with_caption_provider_version(tmp_path):
    database = tmp_path / "dataset.db"
    DatasetManifest(database)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE dataset_items DROP COLUMN caption_provider_version")
        connection.execute("ALTER TABLE dataset_items DROP COLUMN cleanup_provider_version")
        connection.execute("ALTER TABLE dataset_items DROP COLUMN analysis_provider_version")
        connection.execute("ALTER TABLE dataset_items DROP COLUMN crop_provider_version")
        connection.execute("ALTER TABLE dataset_items DROP COLUMN analysis_json")
        connection.execute("ALTER TABLE dataset_items DROP COLUMN crop_json")
        connection.execute("ALTER TABLE dataset_items DROP COLUMN cleanup_verifier_version")
        connection.execute("ALTER TABLE dataset_items DROP COLUMN cleanup_verification_status")
        connection.execute("ALTER TABLE dataset_items DROP COLUMN cleanup_verification_json")
        connection.execute("ALTER TABLE dataset_items DROP COLUMN review_status")
        connection.execute("ALTER TABLE dataset_items DROP COLUMN naming_sequence")
        connection.execute("ALTER TABLE dataset_items DROP COLUMN output_naming_mode")
        connection.execute("ALTER TABLE dataset_items DROP COLUMN lora_name")

    DatasetManifest(database)
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(dataset_items)")}
    assert "caption_provider_version" in columns
    assert "cleanup_provider_version" in columns
    assert "analysis_provider_version" in columns
    assert "crop_provider_version" in columns
    assert "analysis_json" in columns
    assert "crop_json" in columns
    assert "cleanup_verifier_version" in columns
    assert "cleanup_verification_status" in columns
    assert "cleanup_verification_json" in columns
    assert "review_status" in columns
    assert "naming_sequence" in columns
    assert "output_naming_mode" in columns
    assert "lora_name" in columns


def test_force_rebuild_revision_prevents_repeated_queue_rewrites(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "one.png")
    DatasetEngine(source, destination, profile()).run("resume")

    try:
        DatasetEngine(source, destination, profile()).run("force_rebuild")
    except ValueError as error:
        assert "positive rebuild revision" in str(error)
    else:
        raise AssertionError("force_rebuild without a revision should fail safely")

    first = DatasetEngine(
        source,
        destination,
        profile(),
        force_rebuild_revision=1,
    ).run("force_rebuild")
    assert first["processed_this_run"] == 1
    assert first["force_rebuild_reset_items"] == 1

    repeated_queue = DatasetEngine(
        source,
        destination,
        profile(),
        force_rebuild_revision=1,
    ).run("force_rebuild")
    assert repeated_queue["processed_this_run"] == 0
    assert repeated_queue["force_rebuild_reset_items"] == 0

    next_revision = DatasetEngine(
        source,
        destination,
        profile(),
        force_rebuild_revision=2,
    ).run("force_rebuild")
    assert next_revision["processed_this_run"] == 1
    assert next_revision["force_rebuild_reset_items"] == 1
