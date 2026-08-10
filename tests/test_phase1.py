import sqlite3
from pathlib import Path

from PIL import Image

from lora_dataset.engine import DatasetEngine
from lora_dataset.path_utils import is_within, normalized_path
from lora_dataset.profile import DatasetProfileRegistry
from lora_dataset.provider_images import scale_for_provider
from lora_dataset.sidecar import DatasetSidecarWriter, sidecar_stem


def make_image(path, color=(20, 40, 60), size=(32, 24)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def profile(trigger="subject_token", overrides=None):
    return DatasetProfileRegistry().recipe("Krea 2", "Character", trigger, overrides)


def test_sidecar_writer_preserves_basename_and_has_no_numbered_duplicates(tmp_path):
    writer = DatasetSidecarWriter()
    result = writer.write("first", r"H:\input\great photo.webp", tmp_path, "overwrite")
    target = Path(result["path"])
    assert target.name == "great photo.txt"
    assert target.read_text(encoding="utf-8") == "first"

    skipped = writer.write("second", "great photo.webp", tmp_path, "skip")
    assert skipped["status"] == "skipped"
    assert target.read_text(encoding="utf-8") == "first"
    writer.write("third", "great photo.webp", tmp_path, "overwrite")
    assert target.read_text(encoding="utf-8") == "third"
    assert list(tmp_path.glob("great photo*.txt")) == [target]


def test_sidecar_stem_accepts_windows_and_posix_paths_on_any_host():
    assert sidecar_stem(r"H:\input\great photo.webp") == "great photo"
    assert sidecar_stem("/mnt/input/great photo.webp") == "great photo"


def test_paths_are_normalized_without_cross_root_relative_math(tmp_path):
    child = normalized_path("child", base=tmp_path)
    assert child.is_absolute()
    assert child == (tmp_path / "child").resolve()
    assert is_within(child, tmp_path)
    assert not is_within(tmp_path, child)


def test_profile_keeps_recipe_and_custom_caption_instructions():
    recipe = profile(overrides={"additional_caption_instructions": "Focus on the visible tool."})
    assert recipe["profile_id"] == "krea2:character"
    assert recipe["settings"]["crop_strategy"] == "identity_preserving"
    assert recipe["settings"]["additional_caption_instructions"] == "Focus on the visible tool."
    assert recipe["profile_version"]


def test_profile_registry_is_scoped_to_krea2_and_anima():
    registry = DatasetProfileRegistry()
    assert registry.models == ("krea2", "anima")
    assert registry.display_name("krea2") == "Krea 2"
    assert registry.display_name("anima") == "Anima"


def test_nanogpt_image_input_is_limited_to_one_megapixel():
    image = Image.new("RGB", (2000, 1000))
    scaled = scale_for_provider(image, "NanoGPT")
    assert scaled.width * scaled.height <= 1_000_000
    assert scaled.width / scaled.height == 2
    assert scale_for_provider(image, "Ollama") is image


def test_engine_stops_and_resumes_without_duplicate_outputs(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    for index in range(3):
        make_image(source / f"image {index}.png", color=(index, 20, 30))
    (source / "image 0.txt").write_text("standing outdoors", encoding="utf-8")

    first = DatasetEngine(source, destination, profile()).run("resume", max_items=2)
    assert first["complete"] == 2
    assert first["pending"] == 1
    assert not first["watermark_audit_complete"]
    assert not first["training_ready"]
    assert len(list((destination / "dataset").glob("*.png"))) == 2
    assert len(list((destination / "dataset").glob("*.txt"))) == 2

    second = DatasetEngine(source, destination, profile()).run("resume")
    assert second["complete"] == 3
    assert second["pending"] == 0
    assert len(list((destination / "dataset").glob("*.png"))) == 3
    assert len(list((destination / "dataset").glob("*.txt"))) == 3
    assert not list((destination / "dataset").glob("*_1.*"))
    assert (destination / "dataset" / "image 0.txt").read_text(encoding="utf-8") == "subject_token, standing outdoors"

    third = DatasetEngine(source, destination, profile()).run("resume")
    assert third["processed_this_run"] == 0
    assert third["complete"] == 3


def test_interrupted_processing_item_is_reclaimed_on_resume(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "one.jpg")
    engine = DatasetEngine(source, destination, profile())
    engine.synchronize()
    engine.manifest.prepare_run("resume")
    claimed = engine.manifest.claim_next()
    assert claimed["source_relative_path"] == "one.jpg"

    with sqlite3.connect(engine.manifest.path) as connection:
        status = connection.execute("SELECT status FROM dataset_items").fetchone()[0]
    assert status == "processing"

    result = DatasetEngine(source, destination, profile()).run("resume")
    assert result["complete"] == 1
    assert result["failed"] == 0
    assert (destination / "dataset" / "one.png").is_file()
    assert (destination / "dataset" / "one.txt").is_file()


def test_basename_collisions_are_resolved_deterministically(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "a" / "photo.jpg", color=(1, 2, 3))
    make_image(source / "b" / "photo.png", color=(4, 5, 6))

    engine = DatasetEngine(source, destination, profile())
    result = engine.run("resume")
    assert result["complete"] == 2
    image_names = sorted(path.name for path in (destination / "dataset").iterdir() if path.suffix != ".txt")
    caption_names = sorted(path.name for path in (destination / "dataset").glob("*.txt"))
    assert "photo.png" in image_names
    assert any(name.startswith("photo--") and name.endswith(".png") for name in image_names)
    assert "photo.txt" in caption_names
    assert any(name.startswith("photo--") for name in caption_names)

    rerun = DatasetEngine(source, destination, profile()).run("resume")
    assert rerun["processed_this_run"] == 0
    assert image_names == sorted(path.name for path in (destination / "dataset").iterdir() if path.suffix != ".txt")


def test_changed_source_is_reprocessed_at_the_same_output_path(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    image_path = source / "sample.png"
    make_image(image_path, color=(1, 2, 3))
    engine = DatasetEngine(source, destination, profile())
    assert engine.run("resume")["complete"] == 1
    first_hash = engine.manifest.records()[0]["source_hash"]

    make_image(image_path, color=(200, 100, 50))
    resumed = DatasetEngine(source, destination, profile())
    result = resumed.run("resume")
    record = resumed.manifest.records()[0]
    assert result["processed_this_run"] == 1
    assert record["source_hash"] != first_hash
    assert Path(record["output_image_path"]).name == "sample.png"
    assert record["attempts"] == 2
    assert record["normalization_status"] == "converted_png"


def test_validator_flags_orphan_sidecars(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    make_image(source / "sample.png")
    engine = DatasetEngine(source, destination, profile())
    engine.run("resume")
    orphan = destination / "dataset" / "orphan.txt"
    orphan.write_text("unused", encoding="utf-8")
    report = engine.validator.validate_dataset(engine.dataset_directory, engine.manifest.records(), engine.profile)
    assert not report["valid"]
    assert str(orphan.resolve()) in report["orphan_captions"]


def test_removed_source_outputs_are_quarantined_for_review(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    image_path = source / "removed later.png"
    make_image(image_path)
    engine = DatasetEngine(source, destination, profile())
    engine.run("resume")
    image_path.unlink()

    result = DatasetEngine(source, destination, profile()).run("resume")
    assert result["total"] == 0
    assert result["inactive"] == 1
    assert result["quarantined_inactive_outputs"] == 2
    assert not (destination / "dataset" / "removed later.png").exists()
    assert not (destination / "dataset" / "removed later.txt").exists()
    quarantined = list((destination / "review" / "inactive").rglob("*"))
    assert any(path.name == "removed later.png" for path in quarantined)
    assert any(path.name == "removed later.txt" for path in quarantined)


def test_collision_mapping_changes_do_not_leave_stale_dataset_files(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    later = source / "b" / "photo.png"
    make_image(later, color=(1, 2, 3))
    DatasetEngine(source, destination, profile()).run("resume")
    assert (destination / "dataset" / "photo.png").is_file()

    earlier = source / "a" / "photo.jpg"
    make_image(earlier, color=(4, 5, 6))
    result = DatasetEngine(source, destination, profile()).run("resume")
    assert result["complete"] == 2
    assert result["processed_this_run"] == 1
    assert result["quarantined_remapped_outputs"] == 0
    dataset_files = {path.name for path in (destination / "dataset").iterdir()}
    assert "photo.png" in dataset_files
    assert "photo.txt" in dataset_files
    assert any(name.startswith("photo--") and name.endswith(".png") for name in dataset_files)
    assert any(name.startswith("photo--") and name.endswith(".txt") for name in dataset_files)
    assert len(dataset_files) == 4

    earlier.unlink()
    removed = DatasetEngine(source, destination, profile()).run("resume")
    assert removed["complete"] == 1
    assert removed["quarantined_inactive_outputs"] == 0
    assert {path.name for path in (destination / "dataset").iterdir()} == {
        "photo.png", "photo.txt"
    }


def test_non_png_sources_are_normalized_without_modifying_originals(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "project"
    jpeg = source / "camera photo.jpeg"
    webp = source / "transparent.webp"
    make_image(jpeg, color=(100, 120, 140), size=(48, 32))
    source.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (24, 20), (10, 20, 30, 80)).save(webp)
    jpeg_bytes = jpeg.read_bytes()
    webp_bytes = webp.read_bytes()

    result = DatasetEngine(source, destination, profile()).run("resume")
    assert result["complete"] == 2
    assert jpeg.read_bytes() == jpeg_bytes
    assert webp.read_bytes() == webp_bytes
    with Image.open(destination / "dataset" / "camera photo.png") as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (48, 32)
    with Image.open(destination / "dataset" / "transparent.png") as image:
        assert image.format == "PNG"
        assert image.mode == "RGBA"
        assert image.size == (24, 20)
