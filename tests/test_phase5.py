import json
import shutil

import numpy as np
from PIL import Image

from lora_dataset.engine import DatasetEngine
from lora_dataset.intelligence import build_dataset_report
from lora_dataset.manifest import DatasetManifest
from lora_dataset.nodes import DatasetRunSummaryNode, NODE_CLASS_MAPPINGS
from lora_dataset.profile import DatasetProfileRegistry


def profile():
    return DatasetProfileRegistry().recipe("Krea 2", "Character", "subject_token")


def make_image(path, color=(40, 80, 120), size=(640, 640)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def make_gradient(path, changed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.linspace(0, 255, 640, dtype=np.uint8)
    array = np.stack([
        np.broadcast_to(values, (640, 640)),
        np.broadcast_to(values[:, None], (640, 640)),
        np.full((640, 640), 96, dtype=np.uint8),
    ], axis=2)
    if changed:
        array[10, 10] = [255, 0, 255]
    Image.fromarray(array, mode="RGB").save(path)


def test_exact_duplicate_is_excluded_and_reported_without_processing_twice(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_gradient(source / "a.png")
    shutil.copyfile(source / "a.png", source / "b.png")

    engine = DatasetEngine(source, destination, profile())
    result = engine.run("resume")

    assert result["total"] == 2
    assert result["eligible"] == 1
    assert result["complete"] == 1
    assert result["excluded"] == 1
    assert result["processed_this_run"] == 1
    assert result["exact_duplicates_excluded_this_run"] == 1
    assert result["dataset_report"]["duplicates"]["exact_excluded_count"] == 1
    assert result["issues"][0]["stage"] == "duplicate_detection"
    assert result["issues"][0]["image"] == "b.png"
    assert "Exact duplicate of a.png" in result["issues"][0]["reason"]
    assert (destination / "dataset" / "a.png").is_file()
    assert not (destination / "dataset" / "b.png").exists()
    evidence = list((destination / "review" / "duplicates" / "exact").rglob("duplicate.json"))
    assert len(evidence) == 1
    assert json.loads(evidence[0].read_text(encoding="utf-8"))["duplicate_of"] == "a.png"
    assert (destination / "reports" / "dataset_report.json").is_file()

    resumed = engine.run("resume")
    assert resumed["processed_this_run"] == 0
    assert resumed["exact_duplicates_excluded_this_run"] == 0

    (source / "a.png").unlink()
    promoted = DatasetEngine(source, destination, profile()).run("resume")
    assert promoted["exact_duplicates_reinstated_this_run"] == 1
    assert promoted["processed_this_run"] == 1
    assert promoted["complete"] == 1
    assert promoted["excluded"] == 0
    assert promoted["audit_valid"]
    assert resumed["excluded"] == 1


def test_near_duplicates_and_quality_warnings_feed_summary_report(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_gradient(source / "first.png")
    make_gradient(source / "second.png", changed=True)
    make_image(source / "tiny.png", color=(12, 12, 12), size=(96, 96))

    result = DatasetEngine(source, destination, profile()).run("resume")
    report = result["dataset_report"]

    assert report["duplicates"]["near_duplicate_group_count"] >= 1
    flattened = [
        image
        for group in report["duplicates"]["near_duplicate_groups"]
        for image in group["images"]
    ]
    assert "first.png" in flattened
    assert "second.png" in flattened
    tiny = next(item for item in report["quality"]["warning_items"] if item["image"] == "tiny.png")
    assert "low_resolution" in tiny["warnings"]
    assert "low_megapixels" in tiny["warnings"]
    assert report["assessment"] == "NOT_READY"

    rendered = DatasetRunSummaryNode().summarize(json.dumps(result))
    summary, issues_json, report_json = rendered["result"]
    assert "DATASET REPORT" in summary
    assert "Near-duplicate review groups:" in summary
    assert "Quality warnings:" in summary
    assert "tiny.png" in summary
    assert json.loads(issues_json) == []
    assert json.loads(report_json)["duplicates"]["near_duplicate_group_count"] >= 1


def test_numbered_lora_naming_is_stable_and_renames_without_recaptioning(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "a.png", color=(10, 20, 30))
    make_image(source / "b.png", color=(60, 70, 80))

    engine = DatasetEngine(
        source,
        destination,
        profile(),
        output_naming_mode="lora_name_numbered",
        lora_name="taarna",
    )
    first = engine.run("resume")
    assert first["processed_this_run"] == 2
    assert (destination / "dataset" / "taarna_0001.png").is_file()
    assert (destination / "dataset" / "taarna_0001.txt").is_file()
    assert (destination / "dataset" / "taarna_0002.png").is_file()
    captions_before = {
        path.name: path.read_bytes() for path in (destination / "dataset").glob("*.txt")
    }

    make_image(source / "aa.png", color=(100, 110, 120))
    second = engine.run("resume")
    assert second["processed_this_run"] == 1
    assert (destination / "dataset" / "taarna_0003.png").is_file()
    assert (destination / "dataset" / "taarna_0001.txt").read_bytes() == captions_before[
        "taarna_0001.txt"
    ]
    records = DatasetManifest(second["manifest"]).records()
    sequences = {record["source_relative_path"]: record["naming_sequence"] for record in records}
    assert sequences == {"a.png": 1, "aa.png": 3, "b.png": 2}

    renamed = DatasetEngine(
        source,
        destination,
        profile(),
        output_naming_mode="lora_name_numbered",
        lora_name="Taarna Hero",
    ).run("resume")
    assert renamed["processed_this_run"] == 0
    assert sorted(path.name for path in (destination / "dataset").glob("*.png")) == [
        "Taarna Hero_0001.png",
        "Taarna Hero_0002.png",
        "Taarna Hero_0003.png",
    ]
    assert not list((destination / "dataset").glob("taarna_*.png"))
    assert renamed["dataset_report"]["naming"]["lora_names"] == ["Taarna Hero"]
    assert renamed["dataset_report"]["naming"]["stable_sequence_complete"]
    near_groups = renamed["dataset_report"]["duplicates"]["near_duplicate_groups"]
    numbered_names = {
        image for group in near_groups for image in group["images"]
    }
    assert numbered_names == {
        "Taarna Hero_0001.png",
        "Taarna Hero_0002.png",
        "Taarna Hero_0003.png",
    }
    source_names = {
        image: source_name
        for group in near_groups
        for image, source_name in group["source_images"].items()
    }
    assert source_names == {
        "Taarna Hero_0001.png": "a.png",
        "Taarna Hero_0002.png": "b.png",
        "Taarna Hero_0003.png": "aa.png",
    }

    restored = DatasetEngine(
        source,
        destination,
        profile(),
        output_naming_mode="preserve_source_names",
    ).run("resume")
    assert restored["processed_this_run"] == 0
    assert sorted(path.name for path in (destination / "dataset").glob("*.png")) == [
        "a.png",
        "aa.png",
        "b.png",
    ]


def test_krea_report_adds_1024_handoff_trigger_and_recurring_descriptor_guidance(tmp_path):
    dataset = tmp_path / "dataset"
    captions = [
        "subject_token, a woman with long black hair wearing a red coat beside a window",
        "subject_token, a woman with long black hair wearing a blue dress outdoors",
        "subject_token, a woman with long black hair seated on a sofa in soft light",
        "subject_token, a woman with long black hair wearing a white blouse under sunlight",
    ]
    sizes = [(1024, 1024), (1024, 1024), (800, 800), (800, 800)]
    records = []
    for index, (caption, size) in enumerate(zip(captions, sizes), 1):
        image_path = dataset / f"subject_{index:04d}.png"
        caption_path = dataset / f"subject_{index:04d}.txt"
        make_image(image_path, color=(index * 20, index * 30, index * 40), size=size)
        caption_path.write_text(caption, encoding="utf-8")
        records.append({
            "active": 1,
            "status": "complete",
            "source_relative_path": f"source-{index}.jpg",
            "source_hash": f"hash-{index}",
            "output_image_path": str(image_path),
            "caption_path": str(caption_path),
            "analysis_json": "{}",
            "crop_json": "{}",
            "crop_status": "not_needed_allowed_aspect",
            "output_naming_mode": "lora_name_numbered",
            "naming_sequence": index,
            "lora_name": "subject",
            "profile_version": "older-krea-recipe",
        })

    recipe = profile()
    report = build_dataset_report(records, recipe, training_ready=True)
    handoff = report["training_handoff"]
    assert handoff["training_checkpoint"] == "Krea 2 Raw"
    assert handoff["target_bucket_resolution"] == 1024
    assert handoff["at_or_above_target_area_count"] == 2
    assert handoff["below_target_area_count"] == 2
    descriptors = report["captions"]["recurrence"]["recurring_descriptors"]
    assert "long black hair" in {item["text"] for item in descriptors}
    guidance_codes = {item["code"] for item in report["guidance"]["warnings"]}
    assert "krea_1024_bucket_source_coverage_incomplete" in guidance_codes
    assert "krea_character_recurring_descriptors" in guidance_codes
    assert "krea_caption_recipe_revision_mixed" in guidance_codes
    assert "krea_character_trigger_missing" not in guidance_codes
    assert handoff["outdated_caption_profile_count"] == 4
    assert report["assessment"] == "REVIEW_RECOMMENDED"

    missing_trigger = build_dataset_report(
        records,
        DatasetProfileRegistry().recipe("Krea 2", "Character", ""),
        training_ready=True,
    )
    missing_codes = {item["code"] for item in missing_trigger["guidance"]["warnings"]}
    assert "krea_character_trigger_missing" in missing_codes

    summary = DatasetRunSummaryNode().summarize(json.dumps({
        "training_ready": True,
        "eligible": 4,
        "complete": 4,
        "pending": 0,
        "processing": 0,
        "failed": 0,
        "excluded": 0,
        "audit_valid": True,
        "watermark_audit_complete": True,
        "analysis_audit_complete": True,
        "crop_audit_complete": True,
        "dataset_report": report,
        "issues": [],
    }))["result"][0]
    assert "Training target: Krea 2 Raw, 1024 buckets" in summary
    assert "1024 bucket source coverage: 2/4" in summary
    assert "Recurring caption descriptors (trigger-bleed review):" in summary
    assert "long black hair: 4/4 captions (100.0%)" in summary


def test_phase5_summary_node_is_registered():
    assert "LoraDatasetRunSummary" in NODE_CLASS_MAPPINGS
