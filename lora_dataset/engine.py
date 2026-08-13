import json
import os
import re
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from .analysis import BasicAnalysisProvider
from .captioning import (
    apply_trigger,
    apply_video_trigger,
    build_caption_instruction,
    normalize_caption_for_profile,
    normalize_video_caption_for_profile,
)
from .cleanup_verification import prepare_fidelity_reference
from .intelligence import build_dataset_report
from .manifest import DatasetManifest
from .path_utils import ensure_directory, normalized_path
from .sidecar import DatasetSidecarWriter
from .source import DatasetSource
from .transcription import (
    WhisperTranscriber,
    discover_original_video,
    harvester_start_time,
    transcript_for_window,
)
from .validator import DatasetValidator
from .video import (
    VIDEO_EXTENSIONS,
    VideoOrientationExcluded,
    normalize_video_config,
    prepare_video,
    video_config_version,
)


class DatasetItemExcluded(Exception):
    """A confirmed bad image was removed from the eligible training set."""


def _safe_lora_name(value):
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "").strip())
    cleaned = cleaned.rstrip(" .")
    if not cleaned:
        raise ValueError("lora_name is required when output naming is lora_name_numbered")
    return cleaned


def _cleanup_override_keys(value):
    if isinstance(value, (list, tuple, set)):
        entries = [str(item) for item in value]
    else:
        entries = str(value or "").splitlines()
    keys = set()
    for entry in entries:
        cleaned = entry.strip().strip('"').strip("'")
        if cleaned.startswith("- "):
            cleaned = cleaned[2:].strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        keys.add(cleaned.replace("\\", "/").casefold())
        keys.add(Path(cleaned).name.casefold())
    return keys


class DatasetEngine:
    def __init__(
        self,
        source_directory,
        destination_directory,
        profile,
        recursive=True,
        caption_provider=None,
        caption_provider_version="phase1",
        cleanup_provider=None,
        cleanup_provider_version="none",
        cleanup_verifier=None,
        cleanup_verifier_version="none",
        analysis_provider=None,
        analysis_provider_version="basic-v1",
        crop_provider=None,
        crop_provider_version="none",
        force_rebuild_revision=0,
        output_naming_mode="preserve_source_names",
        lora_name="",
        cleanup_override_images="",
        progress_callback=None,
        interrupt_callback=None,
        media_type="images",
        video_config=None,
    ):
        self.source_directory = normalized_path(source_directory)
        self.destination_directory = normalized_path(destination_directory)
        if self.source_directory == self.destination_directory:
            raise ValueError("Source and destination directories must be different")

        self.profile = profile
        self.media_type = "videos" if str(media_type).casefold() in {"video", "videos"} else "images"
        self.video_config = normalize_video_config(video_config) if self.media_type == "videos" else None
        self.video_config_version = video_config_version(self.video_config) if self.video_config else "none"
        self.caption_provider = caption_provider
        self.caption_provider_version = (
            f"{caption_provider_version}+video-{self.video_config_version}"
            if self.video_config else caption_provider_version
        )
        self.cleanup_provider = cleanup_provider
        self.cleanup_provider_version = cleanup_provider_version
        self.cleanup_verifier = cleanup_verifier
        self.cleanup_verifier_version = cleanup_verifier_version
        self.analysis_provider = analysis_provider or BasicAnalysisProvider()
        self.analysis_provider_version = (
            f"{analysis_provider_version}+video-{self.video_config_version}"
            if self.video_config else analysis_provider_version
        )
        self.crop_provider = crop_provider
        self.crop_provider_version = (
            f"{crop_provider_version}+video-{self.video_config_version}"
            if self.video_config else crop_provider_version
        )
        self.force_rebuild_revision = int(force_rebuild_revision)
        if output_naming_mode not in {"preserve_source_names", "lora_name_numbered"}:
            raise ValueError(f"Unknown output naming mode: {output_naming_mode}")
        self.output_naming_mode = output_naming_mode
        self.lora_name = _safe_lora_name(lora_name) if output_naming_mode == "lora_name_numbered" else ""
        self.cleanup_override_images = _cleanup_override_keys(cleanup_override_images)
        self.progress_callback = progress_callback
        self.interrupt_callback = interrupt_callback
        self.dataset_directory = ensure_directory(self.destination_directory / "dataset")
        self.rejected_directory = ensure_directory(self.destination_directory / "rejected")
        self.review_directory = ensure_directory(self.destination_directory / "review")
        self.manifest_directory = ensure_directory(self.destination_directory / "manifest")
        self.reports_directory = ensure_directory(self.destination_directory / "reports")
        self.transcripts_directory = ensure_directory(self.destination_directory / "transcripts")
        self.manifest = DatasetManifest(self.manifest_directory / "dataset.db")
        self.source = DatasetSource(
            self.source_directory,
            recursive=recursive,
            excluded_directories=[self.destination_directory],
            extensions=VIDEO_EXTENSIONS if self.media_type == "videos" else None,
        )
        self.sidecars = DatasetSidecarWriter()
        self.validator = DatasetValidator()
        self.transcriber = None
        self.original_video_path = None
        if self.video_config and self.video_config["transcribe_audio"]:
            self.transcriber = WhisperTranscriber(
                self.transcripts_directory,
                model_name=self.video_config["whisper_model"],
                language=self.video_config["whisper_language"],
                device=self.video_config["whisper_device"],
                ffmpeg_path=self.video_config["ffmpeg_path"],
            )
            self.original_video_path = discover_original_video(
                self.source_directory, self.video_config["original_video_path"]
            )

    def _cleanup_override_for_record(self, record):
        source_relative = str(record.get("source_relative_path") or "").replace("\\", "/")
        candidates = (
            source_relative,
            Path(source_relative).name,
            Path(str(record.get("output_image_path") or "")).name,
        )
        return next(
            (candidate for candidate in candidates if candidate.casefold() in self.cleanup_override_images),
            "",
        )

    def synchronize(self):
        items = self.source.discover()
        assignments = self._assign_output_paths(items)
        previous = {record["item_id"]: record for record in self.manifest.records(active_only=False)}
        remapped = []
        preserve_mapping_ids = set()
        for item in items:
            record = previous.get(item.item_id)
            if record is None:
                continue
            output_image, caption_path = assignments[item.item_id][:2]
            if record["output_image_path"] != str(output_image) or record["caption_path"] != str(caption_path):
                if self._migrate_output_mapping(record, output_image, caption_path):
                    preserve_mapping_ids.add(item.item_id)
                else:
                    remapped.append(record)
        self.manifest.sync(
            items,
            assignments,
            self.profile,
            self.caption_provider_version,
            self.cleanup_provider_version,
            self.cleanup_verifier_version,
            self.analysis_provider_version,
            self.crop_provider_version,
            self.output_naming_mode,
            self.lora_name,
            preserve_mapping_ids,
        )
        self.quarantined_remapped = self._quarantine_records(remapped, "remapped")
        inactive = [record for record in self.manifest.records(active_only=False) if not record["active"]]
        self.quarantined_inactive = self._quarantine_records(inactive, "inactive")
        return items

    def _assign_output_paths(self, items):
        output_suffix = ".mp4" if self.media_type == "videos" else ".png"
        if self.output_naming_mode == "lora_name_numbered":
            sequences = self.manifest.ensure_naming_sequences(items)
            width = max(4, len(str(max(sequences.values(), default=0))))
            return {
                item.item_id: (
                    self.dataset_directory / f"{self.lora_name}_{sequences[item.item_id]:0{width}d}{output_suffix}",
                    self.dataset_directory / f"{self.lora_name}_{sequences[item.item_id]:0{width}d}.txt",
                    sequences[item.item_id],
                )
                for item in items
            }
        used_stems = set()
        assignments = {}
        for item in items:
            source_name = item.path.name
            source_path = Path(source_name)
            stem = source_path.stem
            output_stem = stem
            if stem.casefold() in used_stems:
                output_stem = f"{stem}--{item.item_id[:8]}"
                while output_stem.casefold() in used_stems:
                    output_stem = f"{output_stem}-x"
            used_stems.add(output_stem.casefold())
            output_image = self.dataset_directory / f"{output_stem}{output_suffix}"
            caption_path = self.dataset_directory / f"{output_stem}.txt"
            assignments[item.item_id] = (output_image, caption_path, None)
        return assignments

    def _migrate_output_mapping(self, record, output_image, caption_path):
        pairs = [
            (Path(record["output_image_path"]), Path(output_image)),
            (Path(record["caption_path"]), Path(caption_path)),
        ]
        for source, target in pairs:
            if source == target or not source.exists():
                continue
            if target.exists():
                return False
        moved = []
        try:
            for source, target in pairs:
                if source == target or not source.is_file():
                    continue
                ensure_directory(target.parent)
                os.replace(source, target)
                moved.append((source, target))
        except Exception:
            for source, target in reversed(moved):
                if target.is_file() and not source.exists():
                    os.replace(target, source)
            return False
        return True

    def run(self, mode="resume", max_items=0):
        trigger = self.profile.get("trigger", "").strip()
        profile_model = self.profile.get("model")
        if self.media_type == "videos" and profile_model != "minimax_h3":
            raise ValueError(
                "Video mode uses the MiniMax H3 caption policy. Select MiniMax H3 in LoRA Dataset Profile."
            )
        if self.media_type == "images" and profile_model == "minimax_h3":
            raise ValueError("MiniMax H3 is a video dataset profile. Select videos in LoRA Dataset Source.")
        if self.profile.get("settings", {}).get("trigger_required", False) and not trigger:
            raise ValueError("This dataset profile requires a trigger word")

        discovered = self.synchronize()
        force_rebuild_reset = self.manifest.prepare_run(mode, self.force_rebuild_revision)
        repaired = self.manifest.reset_missing_outputs()
        self.manifest.update_pending_versions(
            self.profile,
            self.caption_provider_version,
            self.cleanup_provider_version,
            self.cleanup_verifier_version,
            self.analysis_provider_version,
            self.crop_provider_version,
        )
        exact_duplicates_excluded, exact_duplicates_reinstated = self._exclude_exact_duplicates()
        processed = 0
        failures = 0
        excluded = 0
        last_file = ""
        limit = max(0, int(max_items))
        work_total = int(self.manifest.summary().get("pending", 0) or 0)
        self._notify_progress(processed, work_total, "", failures, excluded, "running")

        while limit == 0 or processed < limit:
            if self.interrupt_callback is not None:
                self.interrupt_callback()
            record = self.manifest.claim_next()
            if record is None:
                break
            last_file = record["source_relative_path"]
            try:
                caption_status = self._process_record(record)
                validation = self.validator.validate_record(record, self.profile)
                if not validation["valid"]:
                    raise ValueError(", ".join(validation["errors"]))
                self.manifest.mark_complete(record["item_id"], caption_status)
            except DatasetItemExcluded:
                excluded += 1
            except Exception as error:
                failures += 1
                self.manifest.mark_failed(record["item_id"], error)
            processed += 1
            self._notify_progress(
                processed, work_total, last_file, failures, excluded, "running"
            )

        summary = self.manifest.summary()
        active_records = self.manifest.records()
        all_records = self.manifest.records(active_only=False)
        report = self.validator.validate_dataset(
            self.dataset_directory, active_records, self.profile
        )
        issues = self._collect_issues(active_records)
        inactive_sources = [
            record["source_relative_path"] for record in all_records if not record["active"]
        ]
        result = {
            "media_type": self.media_type,
            "video_config": self.video_config,
            "source_directory": str(self.source_directory),
            "destination_directory": str(self.destination_directory),
            "manifest": str(self.manifest.path),
            "output_naming_mode": self.output_naming_mode,
            "lora_name": self.lora_name,
            "discovered": len(discovered),
            "processed_this_run": processed,
            "failed_this_run": failures,
            "excluded_this_run": excluded + exact_duplicates_excluded,
            "exact_duplicates_excluded_this_run": exact_duplicates_excluded,
            "exact_duplicates_reinstated_this_run": exact_duplicates_reinstated,
            "repaired_missing_outputs": repaired,
            "force_rebuild_reset_items": force_rebuild_reset,
            "quarantined_inactive_outputs": self.quarantined_inactive,
            "quarantined_remapped_outputs": self.quarantined_remapped,
            "last_file": last_file,
            **summary,
            "audit_valid": report["valid"],
            "watermark_audit_complete": report["watermark_audit_complete"],
            "cleanup_verification_status_counts": report["cleanup_verification_status_counts"],
            "cleanup_review_items": report["cleanup_review_items"],
            "cleanup_excluded_items": report["cleanup_excluded_items"],
            "cleanup_override_applied_count": report["cleanup_override_applied_count"],
            "residual_detection_mode": (
                self.cleanup_verifier._residual_detection_mode()
                if hasattr(self.cleanup_verifier, "_residual_detection_mode")
                else "custom" if self.cleanup_verifier is not None else "disabled"
            ),
            "analysis_audit_complete": report["analysis_audit_complete"],
            "crop_audit_complete": report["crop_audit_complete"],
            "analysis_status_counts": report["analysis_status_counts"],
            "crop_status_counts": report["crop_status_counts"],
            "issues": issues,
            "inactive_sources": inactive_sources,
            "training_ready": (
                report["valid"]
                and report["watermark_audit_complete"]
                and report["analysis_audit_complete"]
                and report["crop_audit_complete"]
                and summary["eligible"] > 0
                and summary["complete"] == summary["eligible"]
            ),
        }
        dataset_report = build_dataset_report(all_records, self.profile, result["training_ready"])
        dataset_report_path = self._write_dataset_report(dataset_report)
        result["dataset_report"] = dataset_report
        result["dataset_report_path"] = str(dataset_report_path)
        self._write_report({"run": result, "validation": report, "profile": self.profile})
        self._notify_progress(
            processed, work_total, last_file, failures, excluded, "complete"
        )
        if self.transcriber is not None:
            self.transcriber.close()
        return result

    def _notify_progress(self, processed, total, current_file, failed, excluded, status):
        if self.progress_callback is None:
            return
        try:
            self.progress_callback({
                "status": str(status),
                "processed": int(processed),
                "total": int(total),
                "current_file": str(current_file or ""),
                "failed": int(failed),
                "excluded": int(excluded),
            })
        except Exception:
            pass

    def _exclude_exact_duplicates(self):
        if not self.profile.get("settings", {}).get("exclude_exact_duplicates", True):
            return 0, 0
        groups = {}
        for record in self.manifest.records():
            groups.setdefault(record["source_hash"], []).append(record)
        excluded = 0
        reinstated = 0
        for records in groups.values():
            if len(records) == 1:
                record = records[0]
                if (
                    record.get("status") == "excluded"
                    and record.get("review_status") == "exact_duplicate_excluded"
                ):
                    reinstated += self.manifest.reinstate_exact_duplicate(record["item_id"])
                continue
            non_excluded = [record for record in records if record.get("status") != "excluded"]
            candidates = non_excluded or [
                record for record in records
                if record.get("review_status") != "exact_duplicate_excluded"
            ]
            candidates = candidates or records
            ordered = sorted(
                candidates,
                key=lambda record: (
                    0 if record.get("status") == "complete" else 1,
                    record["source_relative_path"].casefold(),
                    record["item_id"],
                ),
            )
            keeper = ordered[0]
            if keeper.get("review_status") == "exact_duplicate_excluded":
                reinstated += self.manifest.reinstate_exact_duplicate(keeper["item_id"])
            for duplicate in records:
                if duplicate["item_id"] == keeper["item_id"]:
                    continue
                if duplicate.get("status") == "excluded":
                    continue
                review_path = self._route_exact_duplicate(duplicate, keeper)
                reason = (
                    f"Exact duplicate of {keeper['source_relative_path']}; "
                    f"excluded from training: {review_path}"
                )
                self.manifest.mark_excluded(
                    duplicate["item_id"], reason, "exact_duplicate_excluded"
                )
                excluded += 1
        return excluded, reinstated

    def _route_exact_duplicate(self, duplicate, keeper):
        review_path = ensure_directory(
            self.review_directory
            / "duplicates"
            / "exact"
            / f"{duplicate['item_id']}-{duplicate['source_hash'][:12]}"
        )
        for field in ("output_image_path", "caption_path"):
            source = Path(duplicate[field])
            if source.is_file():
                os.replace(source, review_path / source.name)
        evidence = {
            "status": "exact_duplicate_excluded",
            "image": duplicate["source_relative_path"],
            "duplicate_of": keeper["source_relative_path"],
            "source_hash": duplicate["source_hash"],
        }
        evidence_path = review_path / "duplicate.json"
        temporary = evidence_path.with_name(f".{evidence_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, evidence_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return review_path

    def _collect_issues(self, records):
        issues = []
        for record in records:
            if record.get("status") not in {"failed", "excluded"}:
                continue
            cleanup_status = record.get("cleanup_verification_status", "not_requested")
            if cleanup_status not in {"not_requested", "verification_processing", "verified_clean", "skipped_video"}:
                stage = "cleanup_verification"
            elif record.get("review_status") == "exact_duplicate_excluded":
                stage = "duplicate_detection"
            elif str(record.get("watermark_status", "")).endswith("failed"):
                stage = "cleanup"
            elif str(record.get("analysis_status", "")).endswith("failed"):
                stage = "analysis"
            elif str(record.get("crop_status", "")).endswith("failed"):
                stage = "crop"
            else:
                stage = "caption_or_validation"
            if record.get("review_status") == "exact_duplicate_excluded":
                review_path = (
                    self.review_directory
                    / "duplicates"
                    / "exact"
                    / f"{record['item_id']}-{record['source_hash'][:12]}"
                )
            else:
                review_path = (
                    self.review_directory
                    / "cleanup_verification"
                    / f"{record['item_id']}-{record['source_hash'][:12]}-{record['profile_version']}"
                )
            output_path = str(record.get("output_image_path") or "")
            output_image = Path(output_path).name if output_path else ""
            verification = {}
            try:
                parsed = json.loads(record.get("cleanup_verification_json") or "{}")
                if isinstance(parsed, dict):
                    verification = parsed
            except (TypeError, json.JSONDecodeError):
                pass
            residual_detections = []
            raw_detections = verification.get("residual_detections") or []
            if isinstance(raw_detections, list):
                for detection in raw_detections:
                    if not isinstance(detection, dict):
                        continue
                    residual_detections.append({
                        key: detection[key]
                        for key in ("label", "confidence", "bbox", "bbox_normalized")
                        if key in detection
                    })
            issues.append({
                "image": output_image or record["source_relative_path"],
                "source_image": record["source_relative_path"],
                "status": record["status"],
                "stage": stage,
                "reason": record.get("error") or cleanup_status or "Unknown processing error",
                "cleanup_verification_status": cleanup_status,
                "detector_confidence_threshold": verification.get("confidence_threshold"),
                "residual_detection_count": int(
                    verification.get("residual_detection_count", len(residual_detections)) or 0
                ),
                "residual_detections": residual_detections,
                "review_status": record.get("review_status", "not_requested"),
                "review_directory": str(review_path) if review_path.is_dir() else "",
            })
        return issues

    def _quarantine_records(self, records, reason):
        moved = 0
        claimed_paths = {
            os.path.normcase(os.path.abspath(record[field]))
            for record in self.manifest.records()
            for field in ("output_image_path", "caption_path")
        }
        for record in records:
            quarantine = ensure_directory(
                self.review_directory
                / reason
                / f"{record['item_id']}-{record['source_hash'][:12]}-{record['profile_version']}"
            )
            for field in ("output_image_path", "caption_path"):
                source = Path(record[field])
                if not source.is_file():
                    continue
                if os.path.normcase(os.path.abspath(source)) in claimed_paths:
                    continue
                target = quarantine / source.name
                os.replace(source, target)
                moved += 1
        return moved

    def _process_record(self, record):
        if self.media_type == "videos":
            return self._process_video_record(record)
        source_path = Path(record["source_path"])
        output_path = Path(record["output_image_path"])
        ensure_directory(output_path.parent)
        temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with Image.open(source_path) as image:
                image.seek(0)
                normalized = ImageOps.exif_transpose(image)
                has_alpha = "A" in normalized.getbands() or "transparency" in image.info
                normalized = normalized.convert("RGBA" if has_alpha else "RGB")
                save_options = {"format": "PNG", "compress_level": 6}
                if image.info.get("icc_profile"):
                    save_options["icc_profile"] = image.info["icc_profile"]
                normalized.save(temporary, **save_options)
            os.replace(temporary, output_path)
        finally:
            if temporary.exists():
                temporary.unlink()

        fidelity_reference = None
        if self.cleanup_verifier is not None:
            fidelity_reference = prepare_fidelity_reference(output_path, megapixels=1.0)

        if self.cleanup_provider is not None:
            self.manifest.mark_watermark_status(record["item_id"], "cleanup_processing")
            try:
                cleanup_result = self.cleanup_provider.clean(
                    output_path,
                    {
                        "source_path": str(source_path),
                        "profile_id": self.profile.get("profile_id"),
                    },
                )
            except Exception:
                self.manifest.mark_watermark_status(record["item_id"], "cleanup_failed")
                raise
            self.manifest.mark_watermark_status(
                record["item_id"], cleanup_result.get("status", "cleaned_universal")
            )

        if self.cleanup_verifier is not None:
            self.manifest.mark_cleanup_verification(
                record["item_id"], "verification_processing", {}, "not_requested"
            )
            try:
                verification = self.cleanup_verifier.verify(
                    fidelity_reference,
                    output_path,
                    {
                        "source_path": str(source_path),
                        "profile_id": self.profile.get("profile_id"),
                    },
                )
            except Exception as error:
                verification = {
                    "status": "verification_failed",
                    "passed": False,
                    "error": str(error),
                }
                self.manifest.mark_cleanup_verification(
                    record["item_id"],
                    "verification_failed",
                    verification,
                    "cleanup_review_required",
                )
                self.manifest.mark_watermark_status(record["item_id"], "verification_failed")
                review_path = self._route_cleanup_review(record, output_path, verification)
                raise RuntimeError(
                    f"Cleanup verifier failed and requires review: {review_path}"
                ) from error
            verification_status = verification.get("status", "verification_failed")
            override_name = self._cleanup_override_for_record(record)
            if (
                override_name
                and verification_status == "residual_artifact"
                and verification.get("residual_detections")
                and not verification.get("fidelity_failures")
            ):
                verification = dict(verification)
                verification.update({
                    "status": "verified_clean_override",
                    "passed": True,
                    "override_applied": True,
                    "override_matched_name": override_name,
                    "override_original_status": verification_status,
                    "override_reason": "User approved legitimate scene or clothing text",
                })
                verification_status = verification["status"]
            if not verification.get("passed", False):
                self.manifest.mark_cleanup_verification(
                    record["item_id"],
                    verification_status,
                    verification,
                    "cleanup_excluded",
                )
                self.manifest.mark_watermark_status(record["item_id"], verification_status)
                review_path = self._route_cleanup_review(record, output_path, verification)
                self.manifest.mark_excluded(
                    record["item_id"],
                    f"Excluded after cleanup verification ({verification_status}): {review_path}",
                )
                raise DatasetItemExcluded(
                    f"Cleanup verification excluded image ({verification_status}): {review_path}"
                )
            self.manifest.mark_cleanup_verification(
                record["item_id"], "verified_clean", verification, "not_requested"
            )
            self.manifest.mark_watermark_status(record["item_id"], "verified_clean")

        try:
            analysis = self.analysis_provider.analyze(
                output_path,
                self.profile,
                {
                    "source_path": str(source_path),
                    "profile_id": self.profile.get("profile_id"),
                },
            )
        except Exception:
            self.manifest.mark_analysis(record["item_id"], "analysis_failed", {})
            raise
        analysis_status = f"analyzed_{analysis.get('provider', 'unknown')}"
        self.manifest.mark_analysis(record["item_id"], analysis_status, analysis)

        if self.crop_provider is not None:
            self.manifest.mark_crop(record["item_id"], "crop_processing", {})
            try:
                crop_result = self.crop_provider.crop(
                    output_path,
                    analysis,
                    self.profile,
                    {
                        "source_path": str(source_path),
                        "profile_id": self.profile.get("profile_id"),
                    },
                )
            except Exception as error:
                self.manifest.mark_crop(
                    record["item_id"], "crop_failed", {"error": str(error)}
                )
                raise
            self.manifest.mark_crop(
                record["item_id"], crop_result.get("status", "crop_complete"), crop_result
            )

        if self.caption_provider is not None:
            instruction = build_caption_instruction(self.profile)
            caption_context = {
                    "source_path": str(source_path),
                    "dataset_type": self.profile.get("dataset_type"),
                    "profile_id": self.profile.get("profile_id"),
                    "analysis": analysis,
                }
            caption = None
            validation_error = None
            for attempt in range(3):
                current_instruction = instruction
                current_context = dict(caption_context)
                if attempt:
                    current_instruction += self._positive_validation_retry_instruction(attempt)
                    current_context["validation_retry"] = True
                    current_context["validation_retry_attempt"] = attempt
                raw_caption = self.caption_provider.caption(
                    output_path,
                    current_instruction,
                    current_context,
                )
                try:
                    caption = normalize_caption_for_profile(raw_caption, self.profile)
                    break
                except ValueError as error:
                    validation_error = str(error)
                    if attempt >= 2:
                        raise
            if caption is None:
                raise ValueError(validation_error or "Caption validation failed")
            caption = apply_trigger(caption, self.profile)
            caption_status = "generated_after_validation_retry" if validation_error else "generated"
        else:
            source_caption = source_path.with_suffix(".txt")
            if source_caption.is_file():
                caption = source_caption.read_text(encoding="utf-8-sig").strip()
                caption_status = "copied_source"
            else:
                caption = self.profile.get("trigger", "").strip()
                caption_status = "trigger_placeholder"

            with_trigger = apply_trigger(caption, self.profile)
            if with_trigger != caption:
                caption_status = f"{caption_status}_with_trigger"
            caption = with_trigger
        self.sidecars.write(caption, output_path.name, output_path.parent, existing_file="overwrite")
        return caption_status

    def _process_video_record(self, record):
        source_path = Path(record["source_path"])
        output_path = Path(record["output_image_path"])
        try:
            metadata = prepare_video(source_path, output_path, self.video_config)
        except VideoOrientationExcluded as error:
            self.manifest.mark_excluded(record["item_id"], str(error), "orientation_filtered")
            raise DatasetItemExcluded(str(error)) from error
        self.manifest.mark_watermark_status(record["item_id"], "skipped_video")
        self.manifest.mark_cleanup_verification(
            record["item_id"], "skipped_video", {"media_type": "video"}, "not_requested"
        )
        self.manifest.mark_analysis(
            record["item_id"],
            "skipped_video",
            {"provider": "ffprobe", "media_type": "video", "video": metadata},
        )
        self.manifest.mark_crop(
            record["item_id"],
            "cropped_ffmpeg" if self.video_config["resize_mode"] == "crop_to_fill" else "prepared_ffmpeg",
            {"provider": "ffmpeg", "settings": self.video_config, "output": metadata},
        )

        if self.caption_provider is not None:
            instruction = build_caption_instruction(self.profile, media_type="video")
            audio_evidence = self._video_audio_evidence(source_path, metadata)
            caption_context = {
                "source_path": str(source_path),
                "dataset_type": self.profile.get("dataset_type"),
                "profile_id": self.profile.get("profile_id"),
                "media_type": "video",
                "video": metadata,
                "video_config": self.video_config,
                "audio_evidence": audio_evidence,
            }
            caption = None
            validation_error = None
            for attempt in range(3):
                current_instruction = instruction
                current_context = dict(caption_context)
                if attempt:
                    current_instruction += self._positive_validation_retry_instruction(attempt)
                    current_context["validation_retry"] = True
                    current_context["validation_retry_attempt"] = attempt
                raw_caption = self.caption_provider.caption_video(
                    output_path, current_instruction, current_context
                )
                try:
                    caption = normalize_video_caption_for_profile(raw_caption, self.profile)
                    break
                except ValueError as error:
                    validation_error = str(error)
                    if attempt >= 2:
                        raise
            if caption is None:
                raise ValueError(validation_error or "Video caption validation failed")
            caption = apply_video_trigger(caption, self.profile)
            caption_status = "generated_after_validation_retry" if validation_error else "generated"
        else:
            source_caption = source_path.with_suffix(".txt")
            if source_caption.is_file():
                caption = source_caption.read_text(encoding="utf-8-sig").strip()
                caption_status = "copied_source"
            else:
                caption = self.profile.get("trigger", "").strip()
                caption_status = "trigger_placeholder"
            with_trigger = apply_video_trigger(caption, self.profile)
            if with_trigger != caption:
                caption_status = f"{caption_status}_with_trigger"
            caption = with_trigger
        self.sidecars.write(caption, output_path.name, output_path.parent, existing_file="overwrite")
        return caption_status

    @staticmethod
    def _positive_validation_retry_instruction(attempt):
        if attempt <= 1:
            return (
                "\n\nGenerate a fresh caption in affirmative language. State present subjects, "
                "objects, actions, camera movement, sounds, and dialogue using the required format."
            )
        return (
            "\n\nCreate a concise replacement caption. Use affirmative sentences naming present "
            "visual events and chronological motion. Return only the finished caption."
        )

    def _video_audio_evidence(self, source_path, metadata):
        if self.transcriber is None:
            return ""
        duration = float(metadata.get("duration") or self.video_config["duration"])
        clip_offset = float(self.video_config.get("start_time") or 0.0)
        harvested_start = harvester_start_time(source_path)
        if self.original_video_path is not None and harvested_start is not None:
            target_start = harvested_start + clip_offset
            transcript = self.transcriber.transcribe_window(
                self.original_video_path, target_start, duration
            )
            return transcript_for_window(transcript, target_start, duration)
        transcript = self.transcriber.transcribe(source_path)
        return transcript_for_window(transcript, clip_offset, duration)

    def _route_cleanup_review(self, record, output_path, verification):
        review_path = ensure_directory(
            self.review_directory
            / "cleanup_verification"
            / f"{record['item_id']}-{record['source_hash'][:12]}-{record['profile_version']}"
        )
        for source in (Path(output_path), Path(record["caption_path"])):
            if source.is_file():
                os.replace(source, review_path / source.name)
        evidence_path = review_path / "verification.json"
        temporary = evidence_path.with_name(f".{evidence_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, evidence_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return review_path

    def _write_report(self, payload):
        report_path = self.reports_directory / "phase1_report.json"
        temporary = report_path.with_name(f".{report_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, report_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _write_dataset_report(self, payload):
        report_path = self.reports_directory / "dataset_report.json"
        temporary = report_path.with_name(f".{report_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, report_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return report_path
