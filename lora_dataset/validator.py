from collections import Counter, defaultdict
import json
from pathlib import Path

from PIL import Image

from .captioning import normalize_caption_for_profile
from .sidecar import IMAGE_EXTENSIONS
from .video import VIDEO_EXTENSIONS, probe_video


class DatasetValidator:
    def validate_record(self, record, profile):
        errors = []
        warnings = []
        image_path = Path(record["output_image_path"])
        caption_path = Path(record["caption_path"])
        is_video = image_path.suffix.casefold() in VIDEO_EXTENSIONS
        video_metadata = None

        if not image_path.is_file():
            errors.append("video_missing" if is_video else "image_missing")
            image_dimensions = None
            aspect_ratio = None
        elif is_video:
            try:
                crop = json.loads(record.get("crop_json") or "{}")
                video_config = crop.get("settings") if isinstance(crop, dict) else None
                video_metadata = probe_video(image_path, video_config)
                image_dimensions = [video_metadata["width"], video_metadata["height"]]
                aspect_ratio = round(video_metadata["width"] / video_metadata["height"], 6)
                if min(image_dimensions) < 512:
                    warnings.append("low_resolution_below_512")
                if video_metadata["duration"] <= 0 or video_metadata["fps"] <= 0:
                    errors.append("video_invalid_timing")
                if isinstance(video_config, dict):
                    target_frames = int(video_config.get("target_frame_count") or 0)
                    if target_frames and video_metadata["frames"] != target_frames:
                        errors.append("video_frame_count_mismatch")
                    target_fps = float(video_config.get("fps") or 0)
                    if target_fps and abs(video_metadata["fps"] - target_fps) > 0.01:
                        errors.append("video_fps_mismatch")
                    if (
                        video_config.get("size_strategy") == "normalize_by_orientation"
                        and video_config.get("resize_mode") in {"crop_to_fill", "pad_to_fit", "stretch"}
                    ):
                        prepared = crop.get("output") if isinstance(crop, dict) else {}
                        orientation = (
                            prepared.get("source_orientation")
                            if isinstance(prepared, dict)
                            else None
                        )
                        if orientation not in {"landscape", "portrait"}:
                            orientation = "landscape" if video_metadata["width"] >= video_metadata["height"] else "portrait"
                        expected = [
                            int(video_config[f"{orientation}_width"]),
                            int(video_config[f"{orientation}_height"]),
                        ]
                        if image_dimensions != expected:
                            errors.append("video_orientation_size_mismatch")
            except Exception:
                errors.append("video_corrupt")
                image_dimensions = None
                aspect_ratio = None
        else:
            try:
                with Image.open(image_path) as image:
                    image_dimensions = [image.width, image.height]
                    aspect_ratio = round(image.width / image.height, 6)
                    if min(image.width, image.height) < 512:
                        warnings.append("low_resolution_below_512")
                    image.verify()
            except Exception:
                errors.append("image_corrupt")
                image_dimensions = None
                aspect_ratio = None

        caption = ""
        if not caption_path.is_file():
            errors.append("caption_missing")
        else:
            caption = caption_path.read_text(encoding="utf-8-sig").strip()
            if not caption:
                errors.append("caption_empty")
            else:
                try:
                    normalize_caption_for_profile(caption, profile)
                except ValueError as error:
                    if "negative or absence language" in str(error):
                        errors.append("caption_negative_or_absence_language")
                    else:
                        errors.append("caption_profile_invalid")

        trigger = profile.get("trigger", "").strip()
        trigger_required = profile.get("settings", {}).get("trigger_required", False)
        if trigger_required and trigger and trigger.casefold() not in caption.casefold():
            errors.append("trigger_missing")
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "caption": caption,
            "image_dimensions": image_dimensions,
            "aspect_ratio": aspect_ratio,
            "media_type": "video" if is_video else "image",
            "video": video_metadata,
        }

    def validate_dataset(self, dataset_directory, records, profile):
        dataset_path = Path(dataset_directory)
        active_records = [record for record in records if record.get("active", 1)]
        eligible_records = [
            record for record in active_records if record.get("status") != "excluded"
        ]
        item_results = []
        captions = []
        hashes = defaultdict(list)
        known_images = set()
        known_captions = set()

        for record in eligible_records:
            result = self.validate_record(record, profile)
            result["item_id"] = record["item_id"]
            result["source_relative_path"] = record["source_relative_path"]
            item_results.append(result)
            if result["caption"]:
                captions.append(result["caption"])
            hashes[record["source_hash"]].append(record["source_relative_path"])
            known_images.add(Path(record["output_image_path"]).resolve(strict=False))
            known_captions.add(Path(record["caption_path"]).resolve(strict=False))

        actual_images = {
            path.resolve(strict=False) for path in dataset_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        } if dataset_path.is_dir() else set()
        actual_captions = {
            path.resolve(strict=False) for path in dataset_path.glob("*.txt") if path.is_file()
        } if dataset_path.is_dir() else set()
        duplicate_captions = [caption for caption, count in Counter(captions).items() if count > 1]
        exact_duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
        watermark_status_counts = Counter(
            record.get("watermark_status", "not_requested") for record in active_records
        )
        cleanup_verification_status_counts = Counter(
            record.get("cleanup_verification_status", "not_requested")
            for record in active_records
        )
        clean_watermark_states = {
            "not_detected",
            "removed",
            "verified_clean",
            "cleaned_universal",
            "skipped_video",
        }
        watermark_audit_complete = all(
            record.get("watermark_status") == "skipped_video"
            if Path(record.get("output_image_path") or "").suffix.casefold() in VIDEO_EXTENSIONS
            else (
                record.get("watermark_status") == "verified_clean"
                and record.get("cleanup_verification_status") == "verified_clean"
            )
            if record.get("cleanup_verifier_version", "none") != "none"
            else record.get("watermark_status") in clean_watermark_states
            for record in eligible_records
        )
        cleanup_review_items = sum(
            1
            for record in active_records
            if record.get("review_status", "not_requested") == "cleanup_review_required"
        )
        cleanup_excluded_items = sum(
            1
            for record in active_records
            if record.get("status") == "excluded"
            and record.get("review_status", "not_requested") == "cleanup_excluded"
        )
        cleanup_override_applied_count = 0
        for record in eligible_records:
            try:
                verification = json.loads(record.get("cleanup_verification_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                verification = {}
            if isinstance(verification, dict) and verification.get("override_applied") is True:
                cleanup_override_applied_count += 1
        analysis_status_counts = Counter(
            record.get("analysis_status", "not_started") for record in eligible_records
        )
        crop_status_counts = Counter(
            record.get("crop_status", "not_requested") for record in eligible_records
        )
        analysis_audit_complete = all(
            str(record.get("analysis_status", "")).startswith(("analyzed_", "skipped_video"))
            for record in eligible_records
        )
        crop_audit_complete = all(
            record.get("crop_provider_version", "none") == "none"
            or str(record.get("crop_status", "")).startswith(
                ("cropped_", "prepared_", "not_needed_", "skipped_")
            )
            for record in eligible_records
        )
        orientation_counts = Counter()
        aspect_ratio_counts = Counter()
        face_visible_items = 0
        person_visible_items = 0
        for record, result in zip(eligible_records, item_results):
            dimensions = result.get("image_dimensions")
            if dimensions:
                width, height = dimensions
                orientation_counts["square" if width == height else "landscape" if width > height else "portrait"] += 1
            ratio = result.get("aspect_ratio")
            if ratio:
                aspect_ratio_counts[f"{ratio:.3f}"] += 1
            try:
                analysis = json.loads(record.get("analysis_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                analysis = {}
            counts = analysis.get("counts", {})
            if int(counts.get("faces", 0)) > 0:
                face_visible_items += 1
            if int(counts.get("persons", 0)) > 0:
                person_visible_items += 1
        errors = sum(len(result["errors"]) for result in item_results)

        return {
            "valid": errors == 0 and not (actual_images - known_images) and not (actual_captions - known_captions),
            "active_items": len(active_records),
            "items": len(item_results),
            "valid_items": sum(1 for result in item_results if result["valid"]),
            "invalid_items": sum(1 for result in item_results if not result["valid"]),
            "error_count": errors,
            "duplicate_caption_count": len(duplicate_captions),
            "exact_duplicate_groups": exact_duplicate_groups,
            "watermark_status_counts": dict(watermark_status_counts),
            "watermark_audit_complete": watermark_audit_complete,
            "cleanup_verification_status_counts": dict(cleanup_verification_status_counts),
            "cleanup_review_items": cleanup_review_items,
            "cleanup_excluded_items": cleanup_excluded_items,
            "cleanup_override_applied_count": cleanup_override_applied_count,
            "analysis_status_counts": dict(analysis_status_counts),
            "analysis_audit_complete": analysis_audit_complete,
            "crop_status_counts": dict(crop_status_counts),
            "crop_audit_complete": crop_audit_complete,
            "orientation_counts": dict(orientation_counts),
            "aspect_ratio_counts": dict(aspect_ratio_counts),
            "face_visible_items": face_visible_items,
            "person_visible_items": person_visible_items,
            "media_type_counts": dict(Counter(result.get("media_type", "image") for result in item_results)),
            "orphan_images": sorted(str(path) for path in actual_images - known_images),
            "orphan_captions": sorted(str(path) for path in actual_captions - known_captions),
            "item_results": item_results,
        }
