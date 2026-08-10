import json
from pathlib import Path

from PIL import Image

from lora_dataset.analysis import AnalysisProvider, BasicAnalysisProvider
from lora_dataset.captioning import CaptionProvider
from lora_dataset.cropping import ProfileSafeCropProvider
from lora_dataset.engine import DatasetEngine
from lora_dataset.manifest import DatasetManifest
from lora_dataset.nodes import DatasetBuilderNode, NODE_CLASS_MAPPINGS
from lora_dataset.profile import DatasetProfileRegistry


def make_image(path, size=(1600, 900), color=(20, 40, 60)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def profile(dataset_type="Character", overrides=None):
    return DatasetProfileRegistry().recipe("Krea 2", dataset_type, "subject_token", overrides)


def analysis(width=1600, height=900, subjects=None, faces=None):
    subjects = subjects or []
    faces = faces or []
    return {
        "schema_version": 1,
        "provider": "fake",
        "dimensions": {
            "width": width,
            "height": height,
            "aspect_ratio": width / height,
            "orientation": "landscape" if width > height else "portrait",
            "megapixels": width * height / 1_000_000,
            "has_alpha": False,
        },
        "quality": {},
        "subjects": subjects,
        "faces": faces,
        "counts": {
            "objects": len(subjects),
            "persons": sum(1 for item in subjects if item["label"] == "person"),
            "faces": len(faces),
        },
    }


class FakeAnalyzer(AnalysisProvider):
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def analyze(self, image_path, profile_value, context=None):
        with Image.open(image_path) as image:
            self.calls.append({"size": image.size, "context": context})
        return self.payload


class RecordingCaptioner(CaptionProvider):
    def __init__(self):
        self.sizes = []

    def caption(self, image_path, instruction, context=None):
        with Image.open(image_path) as image:
            self.sizes.append(image.size)
        return "a person standing in a visible scene"


def test_basic_analyzer_emits_structured_dimensions_and_quality(tmp_path):
    image_path = tmp_path / "wide.png"
    make_image(image_path, size=(64, 32))
    result = BasicAnalysisProvider().analyze(image_path, profile())

    assert result["schema_version"] == 1
    assert result["dimensions"]["width"] == 64
    assert result["dimensions"]["height"] == 32
    assert result["dimensions"]["orientation"] == "landscape"
    assert set(result["quality"]) == {"brightness_mean", "contrast_stddev", "edge_sharpness"}


def test_character_crop_preserves_detected_person_and_uses_profile_ratio(tmp_path):
    image_path = tmp_path / "character.png"
    make_image(image_path)
    person = {"label": "person", "confidence": 0.9, "bbox": [500, 50, 1100, 850]}
    recipe = profile("Character", {
        "allowed_aspect_ratios": ["1:1"],
        "crop_subject_padding_fraction": 0.0,
        "crop_max_removed_fraction": 0.5,
    })
    cropper = ProfileSafeCropProvider({"minimum_output_dimension": 64})
    result = cropper.crop(image_path, analysis(subjects=[person]), recipe)

    assert result["status"] == "cropped_identity_preserving"
    assert result["output_dimensions"] == [900, 900]
    left, top, right, bottom = result["crop_box"]
    assert left <= 500 and right >= 1100
    assert top <= 50 and bottom >= 850
    with Image.open(image_path) as image:
        assert image.size == (900, 900)


def test_style_crop_refuses_large_composition_loss(tmp_path):
    image_path = tmp_path / "style.png"
    make_image(image_path)
    recipe = profile("Style", {"allowed_aspect_ratios": ["1:1"]})
    result = ProfileSafeCropProvider({"minimum_output_dimension": 64}).crop(
        image_path, analysis(), recipe
    )
    assert result["status"] == "skipped_crop_safety_limits"
    with Image.open(image_path) as image:
        assert image.size == (1600, 900)


def test_concept_crop_refuses_to_cut_required_relationship(tmp_path):
    image_path = tmp_path / "concept.png"
    make_image(image_path)
    subjects = [
        {"label": "person", "confidence": 0.9, "bbox": [20, 100, 300, 850]},
        {"label": "bicycle", "confidence": 0.8, "bbox": [1300, 250, 1580, 850]},
    ]
    recipe = profile("Concept", {
        "allowed_aspect_ratios": ["1:1"],
        "crop_max_removed_fraction": 0.5,
        "crop_subject_padding_fraction": 0.0,
    })
    result = ProfileSafeCropProvider({"minimum_output_dimension": 64}).crop(
        image_path, analysis(subjects=subjects), recipe
    )
    assert result["status"] == "skipped_crop_safety_limits"


def test_engine_analyzes_then_crops_before_captioning_and_persists_metadata(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "sample.jpg")
    person = {"label": "person", "confidence": 0.9, "bbox": [500, 50, 1100, 850]}
    analyzer = FakeAnalyzer(analysis(subjects=[person]))
    cropper = ProfileSafeCropProvider({"minimum_output_dimension": 64})
    captioner = RecordingCaptioner()
    recipe = profile("Character", {
        "allowed_aspect_ratios": ["1:1"],
        "crop_subject_padding_fraction": 0.0,
        "crop_max_removed_fraction": 0.5,
    })

    result = DatasetEngine(
        source,
        destination,
        recipe,
        analysis_provider=analyzer,
        analysis_provider_version="fake-analysis-v1",
        crop_provider=cropper,
        crop_provider_version="profile-crop-v1",
        caption_provider=captioner,
        caption_provider_version="caption-v1",
    ).run("resume")

    assert result["complete"] == 1
    assert result["analysis_audit_complete"]
    assert result["crop_audit_complete"]
    assert captioner.sizes == [(900, 900)]
    record = DatasetManifest(result["manifest"]).records()[0]
    assert record["analysis_status"] == "analyzed_fake"
    assert record["crop_status"] == "cropped_identity_preserving"
    assert json.loads(record["analysis_json"])["counts"]["persons"] == 1
    assert json.loads(record["crop_json"])["output_dimensions"] == [900, 900]


def test_phase3_nodes_are_optional_builder_providers():
    optional = DatasetBuilderNode.INPUT_TYPES()["optional"]
    assert "analysis_provider" in optional
    assert "crop_provider" in optional
    assert "LoraDatasetImageAnalyzer" in NODE_CLASS_MAPPINGS
    assert "LoraDatasetSmartCropProvider" in NODE_CLASS_MAPPINGS
