import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from lora_dataset.app_report import render_dataset_summary
from lora_dataset.engine import DatasetEngine
from lora_dataset.nodes import (
    DatasetAppReportNode,
    DatasetCleanupVerifierNode,
    NODE_CLASS_MAPPINGS,
)
from lora_dataset.profile import DatasetProfileRegistry


def profile():
    return DatasetProfileRegistry().recipe("Krea 2", "Character", "app_subject")


def make_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 640), color).save(path)


def status_payload():
    return {
        "training_ready": True,
        "eligible": 2,
        "complete": 2,
        "pending": 0,
        "processing": 0,
        "failed": 0,
        "excluded": 0,
        "audit_valid": True,
        "watermark_audit_complete": True,
        "analysis_audit_complete": True,
        "crop_audit_complete": True,
        "residual_detection_mode": "trust_klein",
        "dataset_report": {
            "assessment": "GOOD",
            "duplicates": {},
            "quality": {"average_score": 100, "warning_item_count": 0},
            "captions": {"count": 2, "average_words": 60, "minimum_words": 55, "maximum_words": 65},
            "distribution": {"orientation_counts": {"square": 2}, "crop_status_counts": {}},
            "naming": {"mode_counts": {"lora_name_numbered": 2}, "lora_names": ["test"], "numbered_pair_count": 2, "sequence_minimum": 1, "sequence_maximum": 2},
        },
        "issues": [],
    }


def test_app_report_renderer_creates_previewable_png(tmp_path):
    output = render_dataset_summary(
        "TRAINING READY\nEligible pairs: 2/2 complete\n\nDATASET REPORT\nAssessment: GOOD",
        tmp_path / "report.png",
    )
    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width == 1280
        assert image.height >= 720


def test_app_report_node_returns_comfy_temp_image_descriptor(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(get_temp_directory=lambda: str(tmp_path)),
    )
    result = DatasetAppReportNode().display(json.dumps(status_payload()))
    assert "TRAINING READY" in result["result"][0]
    assert "Watermark scan: disabled (trusting Klein)" in result["result"][0]
    descriptor = result["ui"]["images"][0]
    assert descriptor["type"] == "temp"
    assert descriptor["subfolder"] == ""
    assert (tmp_path / descriptor["filename"]).is_file()


def test_engine_reports_automatic_full_run_progress(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "one.png", (20, 40, 60))
    make_image(source / "two.png", (80, 100, 120))
    events = []
    interrupts = []
    result = DatasetEngine(
        source,
        destination,
        profile(),
        progress_callback=events.append,
        interrupt_callback=lambda: interrupts.append(True),
    ).run("resume")

    assert result["processed_this_run"] == 2
    assert events[0]["status"] == "running"
    assert events[0]["processed"] == 0
    assert events[0]["total"] == 2
    assert events[-1]["status"] == "complete"
    assert events[-1]["processed"] == 2
    assert events[-1]["current_file"] == "two.png"
    assert len(interrupts) >= 2


def test_phase6_app_workflow_is_preconfigured_for_official_app_mode():
    workflow_path = Path(__file__).parents[1] / "example_workflows" / "LoRA Dataset Builder.app.json"
    if not workflow_path.is_file():
        workflow_path = Path.cwd() / "example_workflows" / "LoRA Dataset Builder.app.json"
    assert workflow_path.stem.endswith(".app")
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    node_types = {node["type"] for node in workflow["nodes"]}
    assert {
        "LoraDatasetSource",
        "LoraDatasetProfile",
        "LoraDatasetCaptionProvider",
        "LoraDatasetKleinCleanupProvider",
        "LoraDatasetCleanupVerifier",
        "LoraDatasetImageAnalyzer",
        "LoraDatasetSmartCropProvider",
        "LoraDatasetVideoPrep",
        "LoraDatasetBuilder",
        "LoraDatasetRunSummary",
        "LoraDatasetAppReport",
    } <= node_types
    assert workflow["extra"]["linearMode"] is True
    assert workflow["extra"]["linearData"]["outputs"] == [10]
    selected_widgets = {
        (entry[0], entry[1]) for entry in workflow["extra"]["linearData"]["inputs"]
    }
    assert (1, "source_directory") in selected_widgets
    assert (1, "media_type") in selected_widgets
    assert (8, "destination_directory") in selected_widgets
    assert (8, "run_mode") in selected_widgets
    assert (8, "output_naming_mode") in selected_widgets
    assert (8, "cleanup_override_images") in selected_widgets
    assert (3, "caption_provider_models") in selected_widgets
    assert (5, "confidence") in selected_widgets
    assert (5, "residual_detection_mode") in selected_widgets
    verifier_node = next(
        node for node in workflow["nodes"] if node["type"] == "LoraDatasetCleanupVerifier"
    )
    assert verifier_node["widgets_values"][1] == 0.3
    assert verifier_node["widgets_values"][-1] == "trust_klein"
    builder_node = next(
        node for node in workflow["nodes"] if node["type"] == "LoraDatasetBuilder"
    )
    builder_inputs = builder_node["inputs"]
    assert builder_inputs[7]["name"] == "caption_provider"
    assert builder_inputs[8]["name"] == "cleanup_provider"
    assert builder_inputs[9]["name"] == "cleanup_verifier"
    assert builder_inputs[10]["name"] == "analysis_provider"
    assert builder_inputs[11]["name"] == "crop_provider"
    assert builder_inputs[12]["name"] == "video_prep"
    assert builder_inputs[13]["name"] == "cleanup_override_images"
    assert len(workflow["links"]) == workflow["last_link_id"]


def test_caption_model_picker_commits_selection_through_comfy_widget_callbacks():
    script_path = Path(__file__).parents[1] / "js" / "lora_dataset_builder.js"
    script = script_path.read_text(encoding="utf-8")
    assert 'modelSelect.addEventListener("input", applySelectedModel)' in script
    assert 'modelSelect.addEventListener("change", applySelectedModel)' in script
    assert "commitWidgetValue(node, modelNameWidget, modelSelect.value)" in script
    assert "widget.callback?.(value, app.canvas, node, undefined, undefined)" in script
    assert "node.onWidgetChanged?.(widget.name, value, previousValue, widget)" in script


def test_cleanup_verifier_uses_conservative_watermark_confidence_default():
    confidence = DatasetCleanupVerifierNode.INPUT_TYPES()["required"]["confidence"]
    assert confidence[1]["default"] == 0.3
    assert confidence[1]["min"] == 0.05


def test_phase6_app_report_node_is_registered():
    assert NODE_CLASS_MAPPINGS["LoraDatasetAppReport"] is DatasetAppReportNode
