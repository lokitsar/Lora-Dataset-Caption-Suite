import json
import uuid
from pathlib import Path

from .analysis import analysis_config_version, create_analysis_provider
from .app_report import render_dataset_summary
from .captioning import DEFAULT_PROVIDER_URLS, create_caption_provider, provider_config_version
from .cleanup import (
    DEFAULT_KLEIN_CLEANUP_PROMPT,
    cleanup_config_version,
    create_cleanup_provider,
)
from .cleanup_verification import (
    MINIMUM_WATERMARK_CONFIDENCE,
    RESIDUAL_DETECTION_MODES,
    VERIFICATION_SCHEMA_VERSION,
    cleanup_verifier_config_version,
    create_cleanup_verifier,
)
from .engine import DatasetEngine
from .cropping import crop_config_version, create_crop_provider
from .manifest import DatasetManifest
from .profile import DatasetProfileRegistry
from .source import DatasetSource
from .validator import DatasetValidator
from .video import (
    CROP_POSITIONS,
    ENCODER_PRESETS,
    ORIENTATION_FILTERS,
    RESIZE_MODES,
    SIZE_STRATEGIES,
    VIDEO_EXTENSIONS,
    normalize_video_config,
    video_config_version,
)
from .transcription import WHISPER_DEVICES, WHISPER_MODELS


PROFILE_REGISTRY = DatasetProfileRegistry()
MODEL_NAMES = [PROFILE_REGISTRY.display_name(model) for model in PROFILE_REGISTRY.models]
DATASET_TYPES = [name.title() for name in PROFILE_REGISTRY.dataset_types]
CAPTION_BACKENDS = list(DEFAULT_PROVIDER_URLS)


def _comfy_model_dropdown(folder_name, preferred_name):
    """Return registered ComfyUI filenames without persisting an absolute model path."""
    try:
        import folder_paths

        names = list(folder_paths.get_filename_list(folder_name))
    except (ImportError, KeyError):
        names = []
    if preferred_name in names:
        names.remove(preferred_name)
    names.insert(0, preferred_name)
    return names


def _ultralytics_model_dropdown(preferred_name):
    names = set()
    try:
        import folder_paths

        for category, prefix in (("ultralytics_bbox", "bbox/"), ("ultralytics_segm", "segm/")):
            try:
                names.update(prefix + name.replace("\\", "/") for name in folder_paths.get_filename_list(category))
            except KeyError:
                pass
        try:
            names.update(name.replace("\\", "/") for name in folder_paths.get_filename_list("ultralytics"))
        except KeyError:
            pass

        model_roots = set()
        for category in ("diffusion_models", "checkpoints", "vae", "text_encoders", "loras"):
            try:
                registered = folder_paths.get_folder_paths(category)
            except KeyError:
                continue
            model_roots.update(Path(folder).resolve().parent for folder in registered)
        for root in model_roots:
            for kind in ("bbox", "segm"):
                directory = root / "ultralytics" / kind
                if not directory.is_dir():
                    continue
                for path in directory.iterdir():
                    if path.is_file() and path.suffix.casefold() in {".pt", ".onnx"}:
                        names.add(f"{kind}/{path.name}")
    except ImportError:
        pass
    names.discard(preferred_name)
    return [preferred_name] + sorted(names, key=str.casefold)


class DatasetProfileNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_model": (MODEL_NAMES,),
                "dataset_type": (DATASET_TYPES,),
                "trigger": ("STRING", {"default": "", "multiline": False}),
                "additional_caption_instructions": (
                    "STRING",
                    {"default": "", "multiline": True, "placeholder": "Optional instructions for unusual or specific datasets"},
                ),
                "advanced_overrides": (
                    "STRING",
                    {"default": "{}", "multiline": True, "placeholder": "Optional JSON profile overrides"},
                ),
            }
        }

    RETURN_TYPES = ("LORA_DATASET_PROFILE", "STRING")
    RETURN_NAMES = ("profile", "profile_json")
    FUNCTION = "build_profile"
    CATEGORY = "LoRA Dataset Caption Suite"

    def build_profile(self, training_model, dataset_type, trigger, additional_caption_instructions, advanced_overrides):
        try:
            overrides = json.loads(advanced_overrides or "{}")
        except json.JSONDecodeError as error:
            raise ValueError(f"advanced_overrides must be valid JSON: {error}") from error
        if not isinstance(overrides, dict):
            raise ValueError("advanced_overrides must be a JSON object")
        if additional_caption_instructions.strip():
            overrides["additional_caption_instructions"] = additional_caption_instructions.strip()
        profile = PROFILE_REGISTRY.recipe(training_model, dataset_type, trigger, overrides)
        return (profile, json.dumps(profile, ensure_ascii=False, indent=2))


class DatasetSourceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_directory": ("STRING", {"default": "", "multiline": False}),
                "recursive": ("BOOLEAN", {"default": True}),
                "media_type": (["images", "videos"],),
            }
        }

    RETURN_TYPES = ("LORA_DATASET_SOURCE", "STRING")
    RETURN_NAMES = ("source", "status")
    FUNCTION = "discover"
    CATEGORY = "LoRA Dataset Caption Suite"

    def discover(self, source_directory, recursive, media_type="images"):
        media_type = "videos" if media_type == "videos" else "images"
        source = DatasetSource(
            source_directory,
            recursive=recursive,
            extensions=VIDEO_EXTENSIONS if media_type == "videos" else None,
        )
        count = len(source.discover())
        config = {
            "source_directory": str(source.root),
            "recursive": bool(recursive),
            "media_type": media_type,
        }
        return (config, f"Discovered {count} supported {media_type} in {source.root}")


class DatasetCaptionProviderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (CAPTION_BACKENDS,),
                "api_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Leave blank to use the selected backend's default URL",
                    },
                ),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "openrouter_key": ("STRING", {"default": "", "multiline": False}),
                "nanogpt_key": ("STRING", {"default": "", "multiline": False}),
                "model_name": ("STRING", {"default": "llama3.2-vision", "multiline": False}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600}),
            }
        }

    RETURN_TYPES = ("LORA_CAPTION_PROVIDER", "STRING")
    RETURN_NAMES = ("caption_provider", "provider_status")
    FUNCTION = "configure"
    CATEGORY = "LoRA Dataset Caption Suite"

    def configure(
        self,
        backend,
        api_url,
        api_key,
        openrouter_key,
        nanogpt_key,
        model_name,
        max_tokens,
        seed,
        timeout,
    ):
        config = {
            "backend": backend,
            "api_url": api_url.strip() or DEFAULT_PROVIDER_URLS[backend],
            "api_key": api_key.strip(),
            "openrouter_key": openrouter_key.strip(),
            "nanogpt_key": nanogpt_key.strip(),
            "model_name": model_name.strip(),
            "max_tokens": int(max_tokens),
            "seed": int(seed),
            "timeout": int(timeout),
        }
        if not config["model_name"]:
            raise ValueError("A caption model name is required")
        status = f"{backend} / {config['model_name']} / config {provider_config_version(config)}"
        return (config, status)


class DatasetImageAnalyzerNode:
    @classmethod
    def INPUT_TYPES(cls):
        subject_models = _ultralytics_model_dropdown("bbox/yolov8s.pt")
        face_models = _ultralytics_model_dropdown("bbox/face_yolov8n.pt")
        return {
            "required": {
                "subject_model": (subject_models, {"default": subject_models[0]}),
                "face_model": (face_models, {"default": face_models[0]}),
                "confidence": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05}),
                "image_size": ("INT", {"default": 640, "min": 320, "max": 1280, "step": 32}),
                "device": (["cpu", "0"],),
            }
        }

    RETURN_TYPES = ("LORA_ANALYSIS_PROVIDER", "STRING")
    RETURN_NAMES = ("analysis_provider", "analysis_status")
    FUNCTION = "configure"
    CATEGORY = "LoRA Dataset Caption Suite"

    def configure(self, subject_model, face_model, confidence, image_size, device):
        config = {
            "provider": "ultralytics",
            "subject_model": subject_model,
            "face_model": face_model,
            "confidence": float(confidence),
            "image_size": int(image_size),
            "device": device,
            "schema_version": 1,
        }
        version = analysis_config_version(config)
        return (config, f"Ultralytics subjects + faces / config {version}")


class DatasetSmartCropProviderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "minimum_output_dimension": (
                    "INT",
                    {"default": 512, "min": 64, "max": 4096, "step": 64},
                ),
            }
        }

    RETURN_TYPES = ("LORA_CROP_PROVIDER", "STRING")
    RETURN_NAMES = ("crop_provider", "crop_status")
    FUNCTION = "configure"
    CATEGORY = "LoRA Dataset Caption Suite"

    def configure(self, minimum_output_dimension):
        config = {
            "provider": "profile_safe_crop",
            "minimum_output_dimension": int(minimum_output_dimension),
            "schema_version": 1,
        }
        version = crop_config_version(config)
        return (config, f"Profile-safe deterministic crop / config {version}")


class DatasetKleinCleanupProviderNode:
    @classmethod
    def INPUT_TYPES(cls):
        diffusion_models = _comfy_model_dropdown(
            "diffusion_models", "flux-2-klein-9b.safetensors"
        )
        text_encoders = _comfy_model_dropdown(
            "text_encoders", "qwen_3_8b_fp8mixed.safetensors"
        )
        vaes = _comfy_model_dropdown("vae", "flux2-vae.safetensors")
        return {
            "required": {
                "diffusion_model": (
                    diffusion_models,
                    {"default": diffusion_models[0]},
                ),
                "text_encoder": (
                    text_encoders,
                    {"default": text_encoders[0]},
                ),
                "vae": (
                    vaes,
                    {"default": vaes[0]},
                ),
                "cleanup_prompt": (
                    "STRING",
                    {"default": DEFAULT_KLEIN_CLEANUP_PROMPT, "multiline": True},
                ),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("LORA_CLEANUP_PROVIDER", "STRING")
    RETURN_NAMES = ("cleanup_provider", "cleanup_status")
    FUNCTION = "configure"
    CATEGORY = "LoRA Dataset Caption Suite"

    def configure(
        self,
        diffusion_model,
        text_encoder,
        vae,
        cleanup_prompt,
        steps,
        seed,
        megapixels,
    ):
        config = {
            "provider": "klein9b_universal_cleanup",
            "diffusion_model": diffusion_model.strip(),
            "text_encoder": text_encoder.strip(),
            "vae": vae.strip(),
            "prompt": cleanup_prompt.strip() or DEFAULT_KLEIN_CLEANUP_PROMPT,
            "steps": int(steps),
            "seed": int(seed),
            "megapixels": float(megapixels),
            "weight_dtype": "default",
        }
        for field in ("diffusion_model", "text_encoder", "vae"):
            if not config[field]:
                raise ValueError(f"Klein cleanup requires {field}")
        version = cleanup_config_version(config)
        return (config, f"Klein 9B universal cleanup / config {version}")


class DatasetCleanupVerifierNode:
    @classmethod
    def INPUT_TYPES(cls):
        models = _ultralytics_model_dropdown("bbox/watermark.pt")
        return {
            "required": {
                "watermark_model": (models, {"default": models[0]}),
                "confidence": (
                    "FLOAT",
                    {
                        "default": MINIMUM_WATERMARK_CONFIDENCE,
                        "min": 0.05,
                        "max": 0.95,
                        "step": 0.05,
                        "tooltip": "Legacy values below 30% are accepted by the workflow UI and migrated to the backend's 30% minimum when residual scanning is enabled.",
                    },
                ),
                "image_size": ("INT", {"default": 640, "min": 320, "max": 1280, "step": 32}),
                "device": (["cpu", "0"],),
                "minimum_structural_similarity": (
                    "FLOAT",
                    {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "maximum_mean_absolute_difference": (
                    "FLOAT",
                    {"default": 0.12, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "maximum_changed_area_fraction": (
                    "FLOAT",
                    {"default": 0.20, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "pixel_change_threshold": (
                    "FLOAT",
                    {"default": 0.30, "min": 0.05, "max": 1.0, "step": 0.05},
                ),
                "maximum_aspect_ratio_delta": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005},
                ),
                "residual_detection_mode": (
                    list(RESIDUAL_DETECTION_MODES),
                    {
                        "default": "trust_klein",
                        "tooltip": "trust_klein skips residual text/watermark detection but keeps image-fidelity verification; verify_residual_watermarks enables the detector.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LORA_CLEANUP_VERIFIER", "STRING")
    RETURN_NAMES = ("cleanup_verifier", "verification_status")
    FUNCTION = "configure"
    CATEGORY = "LoRA Dataset Caption Suite"

    def configure(
        self,
        watermark_model,
        confidence,
        image_size,
        device,
        minimum_structural_similarity,
        maximum_mean_absolute_difference,
        maximum_changed_area_fraction,
        pixel_change_threshold,
        maximum_aspect_ratio_delta,
        residual_detection_mode="trust_klein",
    ):
        config = {
            "provider": "ultralytics_cleanup_verifier",
            "watermark_model": watermark_model,
            "confidence": max(MINIMUM_WATERMARK_CONFIDENCE, float(confidence)),
            "image_size": int(image_size),
            "device": device,
            "minimum_structural_similarity": float(minimum_structural_similarity),
            "maximum_mean_absolute_difference": float(maximum_mean_absolute_difference),
            "maximum_changed_area_fraction": float(maximum_changed_area_fraction),
            "pixel_change_threshold": float(pixel_change_threshold),
            "maximum_aspect_ratio_delta": float(maximum_aspect_ratio_delta),
            "residual_detection_mode": (
                residual_detection_mode
                if residual_detection_mode in RESIDUAL_DETECTION_MODES
                else "trust_klein"
            ),
            "schema_version": VERIFICATION_SCHEMA_VERSION,
        }
        version = cleanup_verifier_config_version(config)
        mode_label = (
            "Klein-trust + fidelity verification"
            if config["residual_detection_mode"] == "trust_klein"
            else "Residual watermark + fidelity verification"
        )
        return (config, f"{mode_label} / config {version}")


class DatasetVideoPrepNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ffmpeg_path": (
                    "STRING",
                    {"default": "", "multiline": False, "placeholder": "Blank = use FFmpeg on PATH"},
                ),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.01}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 2}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 2}),
                "resize_mode": (list(RESIZE_MODES), {"default": "crop_to_fill"}),
                "crop_position": (list(CROP_POSITIONS),),
                "pad_short_video": ("BOOLEAN", {"default": False}),
                "keep_audio": ("BOOLEAN", {"default": True}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51}),
                "encoder_preset": (list(ENCODER_PRESETS),),
                "caption_frames": ("INT", {"default": 8, "min": 2, "max": 32}),
                "caption_megapixels": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 4.0, "step": 0.05}),
                # Appended for saved-workflow compatibility. Exact-frame mode
                # supersedes duration and automatically clone-pads short clips.
                "target_frame_count": ("INT", {
                    "default": 107, "min": 0, "max": 32768, "step": 1,
                    "tooltip": "Exact output frames; 0 uses duration instead. Short clips hold their final frame.",
                }),
                "size_strategy": (list(SIZE_STRATEGIES), {
                    "default": "normalize_by_orientation",
                }),
                "landscape_width": ("INT", {"default": 896, "min": 64, "max": 4096, "step": 2}),
                "landscape_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 2}),
                "portrait_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 2}),
                "portrait_height": ("INT", {"default": 896, "min": 64, "max": 4096, "step": 2}),
                "orientation_filter": (list(ORIENTATION_FILTERS),),
                "transcribe_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use Whisper dialogue and audible transcript cues as additional caption evidence.",
                }),
                "original_video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Recommended: original full movie; blank uses harvester metadata or each clip",
                }),
                "whisper_model": (list(WHISPER_MODELS), {"default": "small.en"}),
                "whisper_language": ("STRING", {"default": "en", "multiline": False}),
                "whisper_device": (list(WHISPER_DEVICES), {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("LORA_VIDEO_PREP", "STRING")
    RETURN_NAMES = ("video_prep", "status")
    FUNCTION = "configure"
    CATEGORY = "LoRA Dataset Caption Suite"

    def configure(self, **kwargs):
        config = normalize_video_config(kwargs)
        ffmpeg = config["ffmpeg_path"] or "FFmpeg on PATH"
        if config["resize_mode"] == "keep_native":
            dimensions = "native source dimensions"
        elif config["size_strategy"] == "normalize_by_orientation":
            dimensions = (
                f"landscape {config['landscape_width']}x{config['landscape_height']}, "
                f"portrait {config['portrait_width']}x{config['portrait_height']}"
            )
        else:
            dimensions = f"{config['width']}x{config['height']}"
        timing = (
            f"exactly {config['target_frame_count']} frames at {config['fps']:g} fps"
            if config["target_frame_count"]
            else f"{config['duration']:g}s at {config['fps']:g} fps"
        )
        status = (
            f"FFmpeg: {ffmpeg} / {timing} / "
            f"{dimensions} {config['resize_mode']} / "
            f"orientation {config['orientation_filter']} / "
            f"{config['caption_frames']} caption frames / "
            f"Whisper {config['whisper_model'] if config['transcribe_audio'] else 'off'} / "
            f"config {video_config_version(config)}"
        )
        return (config, status)


class DatasetBuilderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("LORA_DATASET_SOURCE",),
                "profile": ("LORA_DATASET_PROFILE",),
                "destination_directory": ("STRING", {"default": "", "multiline": False}),
                "run_mode": (
                    ["resume", "reprocess_failed", "force_rebuild"],
                    {
                        "tooltip": "resume = new or source-changed files only; reprocess_failed = those plus failed and cleanup-excluded files; force_rebuild = rewrite every active item once for the selected revision",
                    },
                ),
                "force_rebuild_revision": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "Only for force_rebuild: use a new positive number for each intentional rebuild. Repeated queues of the same revision resume instead of rebuilding again.",
                    },
                ),
                "output_naming_mode": (
                    ["preserve_source_names", "lora_name_numbered"],
                    {
                        "tooltip": "Preserve source basenames, or use a persistent LoRA name plus sequence such as taarna_0001.png/.txt without recaptioning completed pairs.",
                    },
                ),
                "lora_name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Required only for lora_name_numbered output naming.",
                    },
                ),
            },
            "optional": {
                "caption_provider": ("LORA_CAPTION_PROVIDER",),
                "cleanup_provider": ("LORA_CLEANUP_PROVIDER",),
                "cleanup_verifier": ("LORA_CLEANUP_VERIFIER",),
                "analysis_provider": ("LORA_ANALYSIS_PROVIDER",),
                "crop_provider": ("LORA_CROP_PROVIDER",),
                "video_prep": ("LORA_VIDEO_PREP",),
                "cleanup_override_images": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "One explicitly approved source or final PNG filename per line. Overrides residual text/watermark detections only; fidelity failures remain blocked.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status_json", "manifest_path")
    FUNCTION = "build"
    CATEGORY = "LoRA Dataset Caption Suite"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def build(
        self,
        source,
        profile,
        destination_directory,
        run_mode,
        force_rebuild_revision=0,
        output_naming_mode="preserve_source_names",
        lora_name="",
        cleanup_override_images="",
        caption_provider=None,
        cleanup_provider=None,
        cleanup_verifier=None,
        analysis_provider=None,
        crop_provider=None,
        video_prep=None,
        max_items=0,
        unique_id=None,
    ):
        provider = create_caption_provider(caption_provider) if caption_provider else None
        provider_version = provider_config_version(caption_provider) if caption_provider else "phase1"
        cleaner = create_cleanup_provider(cleanup_provider) if cleanup_provider else None
        cleanup_version = cleanup_config_version(cleanup_provider) if cleanup_provider else "none"
        verifier = create_cleanup_verifier(cleanup_verifier) if cleanup_verifier else None
        verifier_version = (
            cleanup_verifier_config_version(cleanup_verifier) if cleanup_verifier else "none"
        )
        analyzer_config = analysis_provider or {"provider": "basic", "schema_version": 1}
        analyzer = create_analysis_provider(analyzer_config)
        analysis_version = analysis_config_version(analyzer_config)
        cropper = create_crop_provider(crop_provider) if crop_provider else None
        crop_version = crop_config_version(crop_provider) if crop_provider else "none"
        progress_bar = None

        def progress_callback(progress):
            nonlocal progress_bar
            total = int(progress.get("total", 0) or 0)
            processed = int(progress.get("processed", 0) or 0)
            if total > 0:
                try:
                    from comfy.utils import ProgressBar

                    if progress_bar is None:
                        progress_bar = ProgressBar(total, node_id=unique_id)
                    progress_bar.update_absolute(processed, total)
                except ImportError:
                    pass
            try:
                from server import PromptServer

                detail = dict(progress)
                detail["node_id"] = unique_id
                PromptServer.instance.send_sync("lora_dataset.progress", detail)
            except (ImportError, AttributeError):
                pass

        def interrupt_callback():
            try:
                from comfy.model_management import throw_exception_if_processing_interrupted

                throw_exception_if_processing_interrupted()
            except ImportError:
                pass

        engine = DatasetEngine(
            source["source_directory"],
            destination_directory,
            profile,
            recursive=source.get("recursive", True),
            caption_provider=provider,
            caption_provider_version=provider_version,
            cleanup_provider=cleaner,
            cleanup_provider_version=cleanup_version,
            cleanup_verifier=verifier,
            cleanup_verifier_version=verifier_version,
            analysis_provider=analyzer,
            analysis_provider_version=analysis_version,
            crop_provider=cropper,
            crop_provider_version=crop_version,
            force_rebuild_revision=force_rebuild_revision,
            output_naming_mode=output_naming_mode,
            lora_name=lora_name,
            cleanup_override_images=cleanup_override_images,
            progress_callback=progress_callback,
            interrupt_callback=interrupt_callback,
            media_type=source.get("media_type", "images"),
            video_config=video_prep,
        )
        try:
            result = engine.run(run_mode, max_items=0)
        except Exception as error:
            progress_callback({
                "status": "error",
                "processed": 0,
                "total": 0,
                "current_file": "",
                "failed": 1,
                "excluded": 0,
                "message": str(error),
            })
            raise
        result["automatic_full_run"] = True
        return (json.dumps(result, ensure_ascii=False, indent=2), str(engine.manifest.path))


class DatasetRunSummaryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "status_json": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect the LoRA Dataset Builder status_json output.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("summary", "issues_json", "report_json")
    FUNCTION = "summarize"
    CATEGORY = "LoRA Dataset Caption Suite"
    OUTPUT_NODE = True

    def summarize(self, status_json):
        try:
            status = json.loads(str(status_json or ""))
        except json.JSONDecodeError as error:
            raise ValueError(f"Builder status_json is not valid JSON: {error}") from error
        if not isinstance(status, dict):
            raise ValueError("Builder status_json must contain a JSON object")

        issues = status.get("issues") or []
        if not isinstance(issues, list):
            issues = []
        eligible = int(status.get("eligible", status.get("total", 0)) or 0)
        complete = int(status.get("complete", 0) or 0)
        failed = int(status.get("failed", 0) or 0)
        excluded = int(status.get("excluded", 0) or 0)
        pending = int(status.get("pending", 0) or 0)
        processing = int(status.get("processing", 0) or 0)
        inactive = int(status.get("inactive", 0) or 0)
        ready = bool(status.get("training_ready", False))
        media_type = str(status.get("media_type") or "images")
        item_label = "videos" if media_type == "videos" else "images"

        if ready:
            verdict = "TRAINING READY"
        elif failed or any(issue.get("status") == "failed" for issue in issues):
            verdict = "ATTENTION REQUIRED"
        else:
            verdict = "NOT READY"
        lines = [
            verdict,
            f"Media: {item_label}",
            f"Eligible pairs: {complete}/{eligible} complete",
            (
                f"This run: {int(status.get('processed_this_run', 0) or 0)} processed, "
                f"{int(status.get('failed_this_run', 0) or 0)} failed, "
                f"{int(status.get('excluded_this_run', 0) or 0)} excluded"
            ),
            f"Current state: {pending} pending, {processing} processing, {failed} failed, {excluded} excluded",
            (
                "Audits: "
                f"dataset={'pass' if status.get('audit_valid') else 'fail'}, "
                f"cleanup={'pass' if status.get('watermark_audit_complete') else 'fail'}, "
                f"analysis={'pass' if status.get('analysis_audit_complete') else 'fail'}, "
                f"crop={'pass' if status.get('crop_audit_complete') else 'fail'}"
            ),
        ]
        residual_mode = str(status.get("residual_detection_mode") or "")
        if residual_mode == "trust_klein":
            lines.append(
                "Watermark scan: disabled (trusting Klein); image-fidelity verification remains enabled."
            )
        elif residual_mode == "verify_residual_watermarks":
            lines.append("Watermark scan: enabled; image-fidelity verification remains enabled.")
        if inactive:
            lines.append(
                f"Inactive history: {inactive} source item(s); this does not block training readiness."
            )
        cleanup_overrides = int(status.get("cleanup_override_applied_count", 0) or 0)
        if cleanup_overrides:
            lines.append(
                f"Cleanup overrides: {cleanup_overrides} explicitly approved image(s); detector evidence retained."
            )

        report = status.get("dataset_report") or {}
        if isinstance(report, dict) and report:
            duplicates = report.get("duplicates") or {}
            quality = report.get("quality") or {}
            captions = report.get("captions") or {}
            distribution = report.get("distribution") or {}
            naming = report.get("naming") or {}
            training_handoff = report.get("training_handoff") or {}
            guidance = report.get("guidance") or {}
            lines.extend([
                "",
                "DATASET REPORT",
                f"Assessment: {report.get('assessment', 'UNKNOWN')}",
                (
                    "Duplicates: "
                    f"{int(duplicates.get('exact_excluded_count', 0) or 0)} exact excluded, "
                    f"{int(duplicates.get('near_duplicate_group_count', 0) or 0)} near-duplicate group(s), "
                    f"{int(duplicates.get('duplicate_caption_group_count', 0) or 0)} duplicate-caption group(s)"
                ),
                (
                    "Quality: "
                    f"average {float(quality.get('average_score', 0) or 0):.1f}/100, "
                    f"{int(quality.get('warning_item_count', 0) or 0)} item(s) with warnings"
                ),
                (
                    "Captions: "
                    f"{int(captions.get('count', 0) or 0)} total, "
                    f"{float(captions.get('average_words', 0) or 0):.1f} average words, "
                    f"range {int(captions.get('minimum_words', 0) or 0)}–{int(captions.get('maximum_words', 0) or 0)}"
                ),
                f"Orientation: {json.dumps(distribution.get('orientation_counts', {}), ensure_ascii=False, sort_keys=True)}",
                f"Crop outcomes: {json.dumps(distribution.get('crop_status_counts', {}), ensure_ascii=False, sort_keys=True)}",
                (
                    "Visibility: "
                    f"faces in {int(distribution.get('face_visible_items', 0) or 0)}, "
                    f"people in {int(distribution.get('person_visible_items', 0) or 0)}"
                ),
            ])
            target_resolution = training_handoff.get("target_bucket_resolution")
            checkpoint = training_handoff.get("training_checkpoint")
            if target_resolution or checkpoint:
                target_parts = []
                if checkpoint:
                    target_parts.append(str(checkpoint))
                if target_resolution:
                    target_parts.append(f"{target_resolution} buckets")
                lines.append("Training target: " + ", ".join(target_parts))
            evaluated = int(training_handoff.get("evaluated_image_count", 0) or 0)
            if target_resolution and evaluated:
                at_target = int(
                    training_handoff.get("at_or_above_target_area_count", 0) or 0
                )
                below_target = int(training_handoff.get("below_target_area_count", 0) or 0)
                lines.append(
                    f"{target_resolution} bucket source coverage: {at_target}/{evaluated} "
                    f"at or above {target_resolution}² pixel area; {below_target} below target"
                )

            guidance_warnings = guidance.get("warnings") or []
            if guidance_warnings:
                lines.append("Training guidance:")
                for warning in guidance_warnings:
                    if isinstance(warning, dict):
                        lines.append(f"- {warning.get('message', warning.get('code', 'Review recommended'))}")

            recurrence = captions.get("recurrence") or {}
            recurring_descriptors = recurrence.get("recurring_descriptors") or []
            if recurring_descriptors:
                lines.append("Recurring caption descriptors (trigger-bleed review):")
                for item in recurring_descriptors[:5]:
                    lines.append(
                        f"- {item.get('text', 'unknown')}: "
                        f"{int(item.get('count', 0) or 0)}/{int(recurrence.get('caption_count', 0) or 0)} "
                        f"captions ({float(item.get('fraction', 0) or 0):.1%})"
                    )
            mode_counts = naming.get("mode_counts") or {}
            if mode_counts:
                lines.append(
                    f"Naming: {json.dumps(mode_counts, ensure_ascii=False, sort_keys=True)}"
                )
                if naming.get("lora_names"):
                    lines.append(
                        "LoRA name: " + ", ".join(str(name) for name in naming["lora_names"])
                    )
                if naming.get("numbered_pair_count"):
                    lines.append(
                        "Stable numbered pairs: "
                        f"{naming['numbered_pair_count']} "
                        f"(sequence {naming.get('sequence_minimum')}–{naming.get('sequence_maximum')})"
                    )

            near_groups = duplicates.get("near_duplicate_groups") or []
            if near_groups:
                lines.append("Near-duplicate review groups:")
                for index, group in enumerate(near_groups, 1):
                    lines.append(
                        f"- Group {index} (distance {group.get('minimum_distance')}): "
                        + ", ".join(group.get("images") or [])
                    )
            warning_items = quality.get("warning_items") or []
            if warning_items:
                lines.append("Quality warnings:")
                for item in warning_items:
                    lines.append(
                        f"- {item.get('image', 'unknown image')}: "
                        f"{', '.join(item.get('warnings') or [])} (score {item.get('score', 0)})"
                    )
            if status.get("dataset_report_path"):
                lines.append(f"Full report: {status['dataset_report_path']}")

        if issues:
            lines.append("")
            lines.append(f"{item_label.title()} needing attention or excluded from training:")
            for issue in issues:
                image_name = issue.get("image") or "unknown image"
                source_name = issue.get("source_image") or ""
                display_name = image_name
                if source_name and source_name != image_name:
                    display_name += f" (source: {source_name})"
                issue_status = str(issue.get("status") or "issue").upper()
                stage = issue.get("stage") or "processing"
                reason = issue.get("reason") or "No reason recorded"
                lines.append(f"- [{issue_status}] {display_name} ({stage}): {reason}")
                confidence_threshold = issue.get("detector_confidence_threshold")
                if isinstance(confidence_threshold, (int, float)):
                    lines.append(f"  Verifier threshold: {confidence_threshold:.1%}")
                detections = issue.get("residual_detections") or []
                for detection in detections[:3] if isinstance(detections, list) else []:
                    if not isinstance(detection, dict):
                        continue
                    label = detection.get("label") or "artifact"
                    confidence = detection.get("confidence")
                    detector_text = f"  Detector: {label}"
                    if isinstance(confidence, (int, float)):
                        detector_text += f" at {confidence:.1%} confidence"
                    box = detection.get("bbox_normalized")
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        try:
                            detector_text += (
                                "; region "
                                f"x {float(box[0]):.1%}–{float(box[2]):.1%}, "
                                f"y {float(box[1]):.1%}–{float(box[3]):.1%}"
                            )
                        except (TypeError, ValueError):
                            pass
                    lines.append(detector_text)
                if isinstance(detections, list) and len(detections) > 3:
                    lines.append(f"  Detector: {len(detections) - 3} additional hit(s) in issues_json")
                if issue.get("review_directory"):
                    lines.append(f"  Review: {issue['review_directory']}")
        else:
            lines.extend(["", f"No active {item_label} errors or exclusions."])

        summary = "\n".join(lines)
        issues_json = json.dumps(issues, ensure_ascii=False, indent=2)
        report_json = json.dumps(report if isinstance(report, dict) else {}, ensure_ascii=False, indent=2)
        return {"ui": {"text": [summary]}, "result": (summary, issues_json, report_json)}


class DatasetAppReportNode:
    @classmethod
    def INPUT_TYPES(cls):
        return DatasetRunSummaryNode.INPUT_TYPES()

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION = "display"
    CATEGORY = "LoRA Dataset Caption Suite"
    OUTPUT_NODE = True

    def display(self, status_json):
        rendered = DatasetRunSummaryNode().summarize(status_json)
        summary = rendered["result"][0]
        ui = {"text": [summary]}
        try:
            import folder_paths

            filename = f"lora_dataset_report_{uuid.uuid4().hex}.png"
            output_path = Path(folder_paths.get_temp_directory()) / filename
            render_dataset_summary(summary, output_path)
            ui["images"] = [{"filename": filename, "subfolder": "", "type": "temp"}]
        except ImportError:
            pass
        return {"ui": ui, "result": (summary,)}


class DatasetValidatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination_directory": ("STRING", {"default": "", "multiline": False}),
                "profile": ("LORA_DATASET_PROFILE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("validation_json",)
    FUNCTION = "validate"
    CATEGORY = "LoRA Dataset Caption Suite"

    def validate(self, destination_directory, profile):
        root = Path(destination_directory).expanduser().resolve(strict=False)
        database = root / "manifest" / "dataset.db"
        if not database.is_file():
            raise FileNotFoundError(f"Dataset manifest does not exist: {database}")
        manifest = DatasetManifest(database)
        report = DatasetValidator().validate_dataset(root / "dataset", manifest.records(), profile)
        return (json.dumps(report, ensure_ascii=False, indent=2),)


NODE_CLASS_MAPPINGS = {
    "LoraDatasetProfile": DatasetProfileNode,
    "LoraDatasetSource": DatasetSourceNode,
    "LoraDatasetCaptionProvider": DatasetCaptionProviderNode,
    "LoraDatasetImageAnalyzer": DatasetImageAnalyzerNode,
    "LoraDatasetSmartCropProvider": DatasetSmartCropProviderNode,
    "LoraDatasetKleinCleanupProvider": DatasetKleinCleanupProviderNode,
    "LoraDatasetCleanupVerifier": DatasetCleanupVerifierNode,
    "LoraDatasetVideoPrep": DatasetVideoPrepNode,
    "LoraDatasetBuilder": DatasetBuilderNode,
    "LoraDatasetRunSummary": DatasetRunSummaryNode,
    "LoraDatasetAppReport": DatasetAppReportNode,
    "LoraDatasetValidator": DatasetValidatorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraDatasetProfile": "LoRA Dataset Profile",
    "LoraDatasetSource": "LoRA Dataset Source",
    "LoraDatasetCaptionProvider": "LoRA Dataset Caption Provider",
    "LoraDatasetImageAnalyzer": "LoRA Dataset Image Analyzer",
    "LoraDatasetSmartCropProvider": "LoRA Dataset Smart Crop",
    "LoraDatasetKleinCleanupProvider": "LoRA Dataset Klein 9B Cleanup",
    "LoraDatasetCleanupVerifier": "LoRA Dataset Cleanup Verifier",
    "LoraDatasetVideoPrep": "LoRA Dataset Video Prep (FFmpeg)",
    "LoraDatasetBuilder": "LoRA Dataset Builder",
    "LoraDatasetRunSummary": "LoRA Dataset Run Summary",
    "LoraDatasetAppReport": "LoRA Dataset App Report",
    "LoraDatasetValidator": "LoRA Dataset Validator",
}
