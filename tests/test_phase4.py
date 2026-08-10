import json
from pathlib import Path

import numpy as np
from PIL import Image

from lora_dataset.captioning import CaptionProvider
from lora_dataset.cleanup_verification import (
    CleanupVerifier,
    UltralyticsCleanupVerifier,
    VERIFICATION_SCHEMA_VERSION,
    fidelity_metrics,
    prepare_fidelity_reference,
)
from lora_dataset.engine import DatasetEngine
from lora_dataset.manifest import DatasetManifest
from lora_dataset.nodes import (
    DatasetBuilderNode,
    DatasetCleanupVerifierNode,
    DatasetRunSummaryNode,
    NODE_CLASS_MAPPINGS,
)
from lora_dataset.profile import DatasetProfileRegistry


def make_pattern(path, size=(128, 96)):
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    array = np.stack(
        [
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            np.full((height, width), 96, dtype=np.uint8),
        ],
        axis=2,
    )
    Image.fromarray(array, mode="RGB").save(path)


def profile():
    return DatasetProfileRegistry().recipe("Krea 2", "Character", "subject_token")


class StaticVerifier(CleanupVerifier):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def verify(self, reference_image, cleaned_image_path, context=None):
        self.calls.append({
            "reference_size": reference_image.size,
            "cleaned_image_path": Path(cleaned_image_path),
            "context": context,
        })
        return dict(self.result)


class FailingVerifier(CleanupVerifier):
    def verify(self, reference_image, cleaned_image_path, context=None):
        raise RuntimeError("detector unavailable")


class SelectiveVerifier(CleanupVerifier):
    def verify(self, reference_image, cleaned_image_path, context=None):
        if Path(cleaned_image_path).stem == "bad":
            return {
                "status": "residual_artifact",
                "passed": False,
                "residual_detection_count": 1,
                "residual_detections": [
                    {"label": "watermark", "confidence": 0.9, "bbox": [10, 10, 100, 50]}
                ],
                "fidelity_metrics": {},
                "fidelity_failures": [],
            }
        return {
            "status": "verified_clean",
            "passed": True,
            "residual_detection_count": 0,
            "residual_detections": [],
            "fidelity_metrics": {},
            "fidelity_failures": [],
        }


class RecordingCaptioner(CaptionProvider):
    def __init__(self):
        self.calls = 0

    def caption(self, image_path, instruction, context=None):
        self.calls += 1
        return "A person stands beside a blue vehicle under warm sunlight."


def test_fidelity_metrics_pass_identical_image_and_flag_rewrite(tmp_path):
    image_path = tmp_path / "pattern.png"
    make_pattern(image_path)
    reference = prepare_fidelity_reference(image_path)
    with Image.open(image_path) as image:
        identical = image.convert("RGB")
    same = fidelity_metrics(reference, identical)
    assert same["structural_similarity"] == 1.0
    assert same["mean_absolute_difference"] == 0.0
    assert same["changed_area_fraction"] == 0.0

    Image.new("RGB", (128, 96), (255, 0, 255)).save(image_path)
    verifier = UltralyticsCleanupVerifier({
        "watermark_model": "none",
        "minimum_structural_similarity": 0.72,
        "maximum_mean_absolute_difference": 0.12,
        "maximum_changed_area_fraction": 0.20,
        "pixel_change_threshold": 0.30,
        "maximum_aspect_ratio_delta": 0.01,
    })
    result = verifier.verify(reference, image_path)
    assert result["status"] == "excessive_change"
    assert not result["passed"]
    assert result["fidelity_failures"]


def test_global_tonal_restoration_is_not_treated_as_structural_damage(tmp_path):
    image_path = tmp_path / "pattern.png"
    make_pattern(image_path)
    reference = prepare_fidelity_reference(image_path)
    restored = np.clip(np.asarray(reference, dtype=np.int16) + 52, 0, 255).astype(np.uint8)
    Image.fromarray(restored, mode="RGB").save(image_path)
    verifier = UltralyticsCleanupVerifier({
        "watermark_model": "none",
        "minimum_structural_similarity": 0.72,
        "maximum_mean_absolute_difference": 0.12,
        "maximum_changed_area_fraction": 0.20,
        "pixel_change_threshold": 0.30,
        "maximum_aspect_ratio_delta": 0.02,
    })

    result = verifier.verify(reference, image_path)

    assert result["status"] == "verified_clean"
    assert result["fidelity_metrics"]["mean_absolute_difference"] > 0.12
    assert result["fidelity_metrics"]["luminance_correlation"] >= 0.90
    assert "mean_absolute_difference_tolerated_as_global_tonal_change" in result[
        "fidelity_observations"
    ]


def test_verified_cleanup_allows_caption_and_training_ready(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_pattern(source / "sample.png", size=(640, 640))
    verifier = StaticVerifier({
        "status": "verified_clean",
        "passed": True,
        "residual_detection_count": 0,
        "residual_detections": [],
        "fidelity_metrics": {},
        "fidelity_failures": [],
    })
    captioner = RecordingCaptioner()
    result = DatasetEngine(
        source,
        destination,
        profile(),
        cleanup_provider=None,
        cleanup_provider_version="none",
        cleanup_verifier=verifier,
        cleanup_verifier_version="verifier-v1",
        caption_provider=captioner,
        caption_provider_version="caption-v1",
    ).run("resume")

    assert result["complete"] == 1
    assert result["watermark_audit_complete"]
    assert result["training_ready"]
    assert captioner.calls == 1
    record = DatasetManifest(result["manifest"]).records()[0]
    assert record["watermark_status"] == "verified_clean"
    assert record["cleanup_verification_status"] == "verified_clean"
    assert json.loads(record["cleanup_verification_json"])["passed"] is True


def test_failed_cleanup_verification_excludes_image_before_captioning(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_pattern(source / "marked.png", size=(640, 640))
    verifier = StaticVerifier({
        "status": "residual_artifact",
        "passed": False,
        "residual_detection_count": 1,
        "residual_detections": [
            {"label": "watermark", "confidence": 0.9, "bbox": [10, 10, 100, 50]}
        ],
        "fidelity_metrics": {},
        "fidelity_failures": [],
    })
    captioner = RecordingCaptioner()
    result = DatasetEngine(
        source,
        destination,
        profile(),
        cleanup_verifier=verifier,
        cleanup_verifier_version="verifier-v1",
        caption_provider=captioner,
        caption_provider_version="caption-v1",
    ).run("resume")

    assert result["failed"] == 0
    assert result["excluded"] == 1
    assert result["eligible"] == 0
    assert result["excluded_this_run"] == 1
    assert result["cleanup_review_items"] == 0
    assert result["cleanup_excluded_items"] == 1
    assert result["watermark_audit_complete"]
    assert not result["training_ready"]
    assert captioner.calls == 0
    assert not (destination / "dataset" / "marked.png").exists()
    review_images = list((destination / "review" / "cleanup_verification").rglob("marked.png"))
    evidence = list((destination / "review" / "cleanup_verification").rglob("verification.json"))
    assert len(review_images) == 1
    assert len(evidence) == 1
    assert json.loads(evidence[0].read_text(encoding="utf-8"))["status"] == "residual_artifact"
    record = DatasetManifest(result["manifest"]).records()[0]
    assert record["status"] == "excluded"
    assert record["review_status"] == "cleanup_excluded"
    assert record["cleanup_verification_status"] == "residual_artifact"


def test_numbered_issue_reports_final_name_source_name_and_detector_evidence(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_pattern(source / "19.png", size=(640, 640))
    verifier = StaticVerifier({
        "status": "residual_artifact",
        "passed": False,
        "residual_detection_count": 1,
        "residual_detections": [
            {
                "label": "watermark",
                "confidence": 0.210587,
                "bbox": [123.075, 293.065, 204.688, 825.238],
                "bbox_normalized": [0.120191, 0.286197, 0.199891, 0.805897],
            }
        ],
        "fidelity_metrics": {"structural_similarity": 0.967931},
        "fidelity_failures": [],
    })
    result = DatasetEngine(
        source,
        destination,
        profile(),
        cleanup_verifier=verifier,
        cleanup_verifier_version="verifier-v1",
        output_naming_mode="lora_name_numbered",
        lora_name="SohannaR-Krea2",
    ).run("resume")

    issue = result["issues"][0]
    assert issue["image"] == "SohannaR-Krea2_0001.png"
    assert issue["source_image"] == "19.png"
    assert issue["residual_detection_count"] == 1
    assert issue["residual_detections"][0]["confidence"] == 0.210587
    assert result["dataset_report"]["review"]["excluded"][0]["image"] == (
        "SohannaR-Krea2_0001.png"
    )
    assert result["dataset_report"]["review"]["excluded"][0]["source_image"] == "19.png"

    summary, issues_json, _report_json = DatasetRunSummaryNode().summarize(
        json.dumps(result)
    )["result"]
    assert "SohannaR-Krea2_0001.png (source: 19.png)" in summary
    assert "watermark at 21.1% confidence" in summary
    assert "x 12.0%–20.0%, y 28.6%–80.6%" in summary
    assert json.loads(issues_json)[0]["source_image"] == "19.png"


def test_excluded_bad_image_does_not_block_remaining_training_dataset(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_pattern(source / "bad.png", size=(640, 640))
    make_pattern(source / "good.png", size=(640, 640))
    with Image.open(source / "good.png") as image:
        distinct = image.copy()
    distinct.putpixel((0, 0), (255, 0, 255))
    distinct.save(source / "good.png")
    captioner = RecordingCaptioner()
    engine = DatasetEngine(
        source,
        destination,
        profile(),
        cleanup_verifier=SelectiveVerifier(),
        cleanup_verifier_version="verifier-v1",
        caption_provider=captioner,
        caption_provider_version="caption-v1",
    )

    result = engine.run("resume")

    assert result["total"] == 2
    assert result["eligible"] == 1
    assert result["complete"] == 1
    assert result["excluded"] == 1
    assert result["failed"] == 0
    assert result["cleanup_excluded_items"] == 1
    assert result["audit_valid"]
    assert result["watermark_audit_complete"]
    assert result["analysis_audit_complete"]
    assert result["crop_audit_complete"]
    assert result["training_ready"]
    assert captioner.calls == 1
    assert (destination / "dataset" / "good.png").is_file()
    assert (destination / "dataset" / "good.txt").is_file()
    assert not (destination / "dataset" / "bad.png").exists()
    assert result["issues"][0]["image"] == "bad.png"
    assert result["issues"][0]["status"] == "excluded"
    assert result["issues"][0]["stage"] == "cleanup_verification"

    rendered = DatasetRunSummaryNode().summarize(json.dumps(result))
    summary = rendered["result"][0]
    issues_json = json.loads(rendered["result"][1])
    assert "TRAINING READY" in summary
    assert "[EXCLUDED] bad.png" in summary
    assert "residual_artifact" in summary
    assert issues_json[0]["image"] == "bad.png"

    resumed = engine.run("resume")
    assert resumed["processed_this_run"] == 0
    assert resumed["excluded"] == 1
    assert resumed["training_ready"]

    downstream_captioner = RecordingCaptioner()
    downstream_change = DatasetEngine(
        source,
        destination,
        profile(),
        cleanup_verifier=SelectiveVerifier(),
        cleanup_verifier_version="verifier-v1",
        caption_provider=downstream_captioner,
        caption_provider_version="caption-v2",
    ).run("resume")
    assert downstream_change["processed_this_run"] == 0
    assert downstream_change["excluded"] == 1
    assert downstream_change["training_ready"]
    assert downstream_captioner.calls == 0

    retry_captioner = RecordingCaptioner()
    reconsidered = DatasetEngine(
        source,
        destination,
        profile(),
        cleanup_verifier=StaticVerifier({
            "status": "verified_clean",
            "passed": True,
            "residual_detection_count": 0,
            "residual_detections": [],
            "fidelity_metrics": {},
            "fidelity_failures": [],
        }),
        cleanup_verifier_version="verifier-v2",
        caption_provider=retry_captioner,
        caption_provider_version="caption-v2",
    ).run("reprocess_failed")
    assert reconsidered["processed_this_run"] == 1
    assert reconsidered["complete"] == 2
    assert reconsidered["excluded"] == 0
    assert reconsidered["training_ready"]
    assert retry_captioner.calls == 1
    assert (destination / "dataset" / "bad.png").is_file()
    assert (destination / "dataset" / "good.png").is_file()


def test_cleanup_verifier_exception_routes_to_review_before_captioning(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_pattern(source / "uncertain.png", size=(640, 640))
    captioner = RecordingCaptioner()
    result = DatasetEngine(
        source,
        destination,
        profile(),
        cleanup_verifier=FailingVerifier(),
        cleanup_verifier_version="verifier-v1",
        caption_provider=captioner,
        caption_provider_version="caption-v1",
    ).run("resume")

    assert result["failed"] == 1
    assert result["cleanup_review_items"] == 1
    assert not result["training_ready"]
    assert captioner.calls == 0
    assert not (destination / "dataset" / "uncertain.png").exists()
    review_images = list((destination / "review" / "cleanup_verification").rglob("uncertain.png"))
    evidence = list((destination / "review" / "cleanup_verification").rglob("verification.json"))
    assert len(review_images) == 1
    assert len(evidence) == 1
    payload = json.loads(evidence[0].read_text(encoding="utf-8"))
    assert payload["status"] == "verification_failed"
    assert "detector unavailable" in payload["error"]


def test_phase4_verifier_is_an_optional_builder_provider():
    optional = DatasetBuilderNode.INPUT_TYPES()["optional"]
    assert "cleanup_verifier" in optional
    assert "LoraDatasetCleanupVerifier" in NODE_CLASS_MAPPINGS
    assert "LoraDatasetRunSummary" in NODE_CLASS_MAPPINGS
    config, _ = DatasetCleanupVerifierNode().configure(
        "bbox/watermark.pt", 0.2, 640, "cpu", 0.72, 0.12, 0.20, 0.30, 0.02
    )
    assert config["schema_version"] == VERIFICATION_SCHEMA_VERSION
    assert config["confidence"] == 0.3
    assert UltralyticsCleanupVerifier({"confidence": 0.2})._confidence_threshold() == 0.3
    assert UltralyticsCleanupVerifier({"confidence": 0.45})._confidence_threshold() == 0.45
