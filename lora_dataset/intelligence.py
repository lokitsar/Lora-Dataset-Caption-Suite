import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


INTELLIGENCE_SCHEMA_VERSION = 2

_DESCRIPTOR_BOUNDARY_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of",
    "on", "or", "the", "to", "under", "with", "while", "he", "she", "they",
    "his", "her", "their", "this", "that",
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _image_reference(record):
    source_image = str(record.get("source_relative_path") or "")
    output_path = str(record.get("output_image_path") or "")
    output_image = Path(output_path).name if output_path else ""
    return {
        "image": output_image or source_image,
        "source_image": source_image,
    }


def difference_hash(image_path, hash_size=8):
    size = max(4, int(hash_size))
    with Image.open(image_path) as image:
        gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    width = (bits.size + 3) // 4
    return f"{value:0{width}x}"


def hamming_distance(first_hash, second_hash):
    return (int(first_hash, 16) ^ int(second_hash, 16)).bit_count()


def _connected_groups(names, pairs):
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for pair in pairs:
        union(pair["first"], pair["second"])
    grouped = defaultdict(list)
    for name in names:
        grouped[find(name)].append(name)
    return [sorted(group, key=str.casefold) for group in grouped.values() if len(group) > 1]


def find_near_duplicate_groups(records, maximum_distance=6, hash_size=8):
    hashed = []
    for record in records:
        image_path = Path(record["output_image_path"])
        if not image_path.is_file():
            continue
        reference = _image_reference(record)
        hashed.append({
            "image": reference["image"],
            "source_image": reference["source_image"],
            "output": str(image_path),
            "hash": difference_hash(image_path, hash_size=hash_size),
            "source_hash": record.get("source_hash", ""),
        })
    pairs = []
    limit = max(0, int(maximum_distance))
    for index, first in enumerate(hashed):
        for second in hashed[index + 1:]:
            if first["source_hash"] and first["source_hash"] == second["source_hash"]:
                continue
            distance = hamming_distance(first["hash"], second["hash"])
            if distance <= limit:
                pairs.append({
                    "first": first["image"],
                    "second": second["image"],
                    "first_source": first["source_image"],
                    "second_source": second["source_image"],
                    "distance": distance,
                })
    groups = _connected_groups([item["image"] for item in hashed], pairs)
    result = []
    for images in groups:
        image_set = set(images)
        group_pairs = [
            pair for pair in pairs
            if pair["first"] in image_set and pair["second"] in image_set
        ]
        result.append({
            "images": images,
            "source_images": {
                item["image"]: item["source_image"]
                for item in hashed
                if item["image"] in image_set
            },
            "minimum_distance": min(pair["distance"] for pair in group_pairs),
            "pairs": sorted(
                group_pairs,
                key=lambda pair: (pair["distance"], pair["first"].casefold(), pair["second"].casefold()),
            ),
        })
    return sorted(result, key=lambda group: group["images"][0].casefold())


def _json_object(value):
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _caption_groups(records):
    captions = defaultdict(list)
    for record in records:
        caption_path = Path(record["caption_path"])
        if not caption_path.is_file():
            continue
        caption = " ".join(caption_path.read_text(encoding="utf-8-sig").split())
        if caption:
            captions[caption.casefold()].append(_image_reference(record)["image"])
    return [
        sorted(images, key=str.casefold)
        for images in captions.values()
        if len(images) > 1
    ]


def _recurring_caption_descriptors(records, trigger="", minimum_fraction=0.60, limit=10):
    caption_tokens = []
    trigger_pattern = re.compile(re.escape(trigger), re.IGNORECASE) if trigger else None
    for record in records:
        caption_path = Path(record["caption_path"])
        if not caption_path.is_file():
            continue
        caption = caption_path.read_text(encoding="utf-8-sig")
        if trigger_pattern is not None:
            caption = trigger_pattern.sub(" ", caption)
        tokens = [
            token.casefold().strip("'-")
            for token in re.findall(r"[\w$'-]+", caption, flags=re.UNICODE)
        ]
        tokens = [token for token in tokens if token]
        if tokens:
            caption_tokens.append(tokens)

    caption_count = len(caption_tokens)
    minimum_count = max(3, int(math.ceil(caption_count * float(minimum_fraction))))
    counts = Counter()
    for tokens in caption_tokens:
        candidates = set()
        for size in (3, 2):
            for index in range(0, len(tokens) - size + 1):
                phrase_tokens = tokens[index:index + size]
                if any(token in _DESCRIPTOR_BOUNDARY_STOPWORDS for token in phrase_tokens):
                    continue
                if all(token.isdigit() for token in phrase_tokens):
                    continue
                candidates.add(" ".join(phrase_tokens))
        counts.update(candidates)

    ranked = [
        {
            "text": phrase,
            "count": count,
            "fraction": round(count / caption_count, 4) if caption_count else 0.0,
        }
        for phrase, count in counts.items()
        if count >= minimum_count
    ]
    ranked.sort(
        key=lambda item: (-item["fraction"], -len(item["text"].split()), item["text"])
    )
    selected = []
    for item in ranked:
        if any(item["text"] in existing["text"] for existing in selected):
            continue
        selected.append(item)
        if len(selected) >= int(limit):
            break
    return {
        "caption_count": caption_count,
        "minimum_fraction": float(minimum_fraction),
        "minimum_count": minimum_count if caption_count else 0,
        "recurring_descriptors": selected,
        "trigger_bleed_review_recommended": bool(selected),
    }


def _training_handoff(profile, quality_items, records):
    settings = profile.get("settings", {})
    preferred = [int(value) for value in settings.get("preferred_resolutions", []) if int(value) > 0]
    target = preferred[0] if preferred else None
    target_pixels = target * target if target else None
    dimensions = [item["dimensions"] for item in quality_items if item.get("dimensions")]
    pixel_counts = [width * height for width, height in dimensions]
    at_target = sum(1 for pixels in pixel_counts if target_pixels and pixels >= target_pixels)
    below_target = len(pixel_counts) - at_target if target_pixels else 0
    current_profile_version = str(profile.get("profile_version") or "")
    profile_version_counts = Counter(
        str(record.get("profile_version"))
        for record in records
        if record.get("profile_version")
    )
    outdated_profile_items = sum(
        count
        for version, count in profile_version_counts.items()
        if current_profile_version and version != current_profile_version
    )
    return {
        "training_checkpoint": settings.get("training_checkpoint"),
        "preferred_resolutions": preferred,
        "target_bucket_resolution": target,
        "target_pixel_area": target_pixels,
        "evaluated_image_count": len(pixel_counts),
        "at_or_above_target_area_count": at_target,
        "below_target_area_count": below_target,
        "target_area_coverage_fraction": (
            round(at_target / len(pixel_counts), 4) if pixel_counts and target_pixels else None
        ),
        "minimum_source_megapixels": (
            round(min(pixel_counts) / 1_000_000, 4) if pixel_counts else None
        ),
        "average_source_megapixels": (
            round(sum(pixel_counts) / len(pixel_counts) / 1_000_000, 4)
            if pixel_counts else None
        ),
        "current_profile_version": current_profile_version,
        "caption_profile_version_counts": dict(profile_version_counts),
        "outdated_caption_profile_count": outdated_profile_items,
        "advisory_only": True,
    }


def _item_quality(record, profile, duplicate_caption_images):
    settings = profile.get("settings", {})
    image_path = Path(record["output_image_path"])
    caption_path = Path(record["caption_path"])
    warnings = []
    score = 100
    dimensions = None
    orientation = "unknown"
    megapixels = 0.0
    if image_path.is_file():
        with Image.open(image_path) as image:
            dimensions = [image.width, image.height]
            orientation = "square" if image.width == image.height else "landscape" if image.width > image.height else "portrait"
            megapixels = (image.width * image.height) / 1_000_000
        minimum_dimension = int(settings.get("quality_minimum_dimension", 512))
        if min(dimensions) < minimum_dimension:
            warnings.append("low_resolution")
            score -= 25
        if megapixels < float(settings.get("quality_minimum_megapixels", 0.35)):
            warnings.append("low_megapixels")
            score -= 10
    else:
        warnings.append("image_missing")
        score -= 60

    analysis = _json_object(record.get("analysis_json"))
    quality = analysis.get("quality", {}) if isinstance(analysis.get("quality"), dict) else {}
    brightness = quality.get("brightness_mean")
    contrast = quality.get("contrast_stddev")
    sharpness = quality.get("edge_sharpness")
    if isinstance(brightness, (int, float)):
        if brightness < float(settings.get("quality_dark_threshold", 22)):
            warnings.append("very_dark")
            score -= 12
        elif brightness > float(settings.get("quality_bright_threshold", 238)):
            warnings.append("very_bright")
            score -= 12
    if isinstance(contrast, (int, float)) and contrast < float(
        settings.get("quality_low_contrast_threshold", 18)
    ):
        warnings.append("low_contrast")
        score -= 10
    if isinstance(sharpness, (int, float)) and sharpness < float(
        settings.get("quality_low_sharpness_threshold", 1.5)
    ):
        warnings.append("low_edge_detail")
        score -= 10

    caption_words = 0
    if caption_path.is_file():
        caption = caption_path.read_text(encoding="utf-8-sig").strip()
        caption_words = len(caption.split())
        if _image_reference(record)["image"] in duplicate_caption_images:
            warnings.append("duplicate_caption")
            score -= 8
        maximum_words = int(settings.get("caption_max_words", 120))
        if caption_words > maximum_words:
            warnings.append("caption_over_profile_limit")
            score -= 15
    else:
        warnings.append("caption_missing")
        score -= 40

    crop = _json_object(record.get("crop_json"))
    removed_fraction = crop.get("removed_area_fraction")
    crop_limit = float(settings.get("crop_max_removed_fraction", 1.0))
    if isinstance(removed_fraction, (int, float)) and crop_limit < 1.0:
        if removed_fraction >= crop_limit * 0.9:
            warnings.append("crop_near_profile_limit")
            score -= 5

    return {
        **_image_reference(record),
        "score": max(0, score),
        "warnings": warnings,
        "dimensions": dimensions,
        "megapixels": round(megapixels, 4),
        "orientation": orientation,
        "caption_words": caption_words,
        "brightness_mean": brightness,
        "contrast_stddev": contrast,
        "edge_sharpness": sharpness,
    }


def build_dataset_report(records, profile, training_ready):
    active_records = [record for record in records if record.get("active", 1)]
    eligible_records = [record for record in active_records if record.get("status") == "complete"]
    excluded_records = [record for record in active_records if record.get("status") == "excluded"]
    failed_records = [record for record in active_records if record.get("status") == "failed"]
    duplicate_caption_groups = _caption_groups(eligible_records)
    duplicate_caption_images = {
        image for group in duplicate_caption_groups for image in group
    }
    maximum_distance = int(
        profile.get("settings", {}).get("near_duplicate_maximum_hamming_distance", 6)
    )
    near_duplicate_groups = find_near_duplicate_groups(
        eligible_records, maximum_distance=maximum_distance
    )
    quality_items = [
        _item_quality(record, profile, duplicate_caption_images)
        for record in eligible_records
    ]
    training_handoff = _training_handoff(profile, quality_items, eligible_records)
    is_krea_character = (
        profile.get("model") == "krea2" and profile.get("dataset_type") == "character"
    )
    caption_recurrence = _recurring_caption_descriptors(
        eligible_records,
        trigger=profile.get("trigger", ""),
    ) if is_krea_character else {
        "caption_count": len(eligible_records),
        "minimum_fraction": 0.60,
        "minimum_count": 0,
        "recurring_descriptors": [],
        "trigger_bleed_review_recommended": False,
    }
    guidance_warnings = []
    if is_krea_character and not str(profile.get("trigger") or "").strip():
        guidance_warnings.append({
            "code": "krea_character_trigger_missing",
            "message": "Krea character training has no trigger; a trigger is recommended to carry stable identity.",
        })
    if (
        profile.get("model") == "krea2"
        and training_handoff.get("below_target_area_count", 0) > 0
    ):
        guidance_warnings.append({
            "code": "krea_1024_bucket_source_coverage_incomplete",
            "message": (
                f"{training_handoff['below_target_area_count']} image(s) are below the "
                f"{training_handoff['target_bucket_resolution']}² preferred source-pixel area; "
                "the trainer may need to upscale them for 1024 buckets."
            ),
        })
    if caption_recurrence["trigger_bleed_review_recommended"]:
        guidance_warnings.append({
            "code": "krea_character_recurring_descriptors",
            "message": (
                "Highly recurring caption descriptors may be learned outside the trigger; "
                "review the recurring-descriptor list for invariant identity traits."
            ),
        })
    if is_krea_character and training_handoff["outdated_caption_profile_count"] > 0:
        guidance_warnings.append({
            "code": "krea_caption_recipe_revision_mixed",
            "message": (
                f"{training_handoff['outdated_caption_profile_count']} caption(s) were generated "
                "with an earlier Krea recipe. Resume will preserve them; use an intentional "
                "revisioned force rebuild only if you want every caption regenerated."
            ),
        })
    scores = [item["score"] for item in quality_items]
    warning_counts = Counter(
        warning for item in quality_items for warning in item["warnings"]
    )
    warning_items = [item for item in quality_items if item["warnings"]]
    orientation_counts = Counter(item["orientation"] for item in quality_items)
    resolution_counts = Counter(
        f"{item['dimensions'][0]}x{item['dimensions'][1]}"
        for item in quality_items if item["dimensions"]
    )
    caption_word_counts = [item["caption_words"] for item in quality_items]
    exact_duplicate_exclusions = [
        {
            **_image_reference(record),
            "reason": record.get("error") or "Exact duplicate excluded",
        }
        for record in excluded_records
        if record.get("review_status") == "exact_duplicate_excluded"
    ]
    review_recommended = bool(
        near_duplicate_groups
        or duplicate_caption_groups
        or guidance_warnings
        or (quality_items and len(warning_items) / len(quality_items) > 0.25)
        or (scores and sum(scores) / len(scores) < 80)
    )
    if not training_ready:
        assessment = "NOT_READY"
    elif review_recommended:
        assessment = "REVIEW_RECOMMENDED"
    else:
        assessment = "GOOD"
    naming_modes = Counter(
        record.get("output_naming_mode", "preserve_source_names")
        for record in eligible_records
    )
    numbered_records = [
        record for record in eligible_records
        if record.get("output_naming_mode") == "lora_name_numbered"
    ]
    sequences = [
        int(record["naming_sequence"])
        for record in numbered_records
        if record.get("naming_sequence") is not None
    ]
    lora_names = sorted({
        record.get("lora_name", "") for record in numbered_records if record.get("lora_name")
    }, key=str.casefold)

    return {
        "schema_version": INTELLIGENCE_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "assessment": assessment,
        "training_ready": bool(training_ready),
        "counts": {
            "active": len(active_records),
            "eligible": len(eligible_records),
            "failed": len(failed_records),
            "excluded": len(excluded_records),
        },
        "duplicates": {
            "exact_excluded_count": len(exact_duplicate_exclusions),
            "exact_excluded": exact_duplicate_exclusions,
            "near_duplicate_group_count": len(near_duplicate_groups),
            "near_duplicate_groups": near_duplicate_groups,
            "duplicate_caption_group_count": len(duplicate_caption_groups),
            "duplicate_caption_groups": duplicate_caption_groups,
            "near_duplicate_maximum_hamming_distance": maximum_distance,
        },
        "quality": {
            "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "minimum_score": min(scores) if scores else 0,
            "maximum_score": max(scores) if scores else 0,
            "warning_item_count": len(warning_items),
            "warning_counts": dict(warning_counts),
            "warning_items": warning_items,
        },
        "captions": {
            "count": len(caption_word_counts),
            "average_words": round(sum(caption_word_counts) / len(caption_word_counts), 2)
            if caption_word_counts else 0.0,
            "minimum_words": min(caption_word_counts) if caption_word_counts else 0,
            "maximum_words": max(caption_word_counts) if caption_word_counts else 0,
            "recurrence": caption_recurrence,
        },
        "training_handoff": training_handoff,
        "guidance": {
            "warning_count": len(guidance_warnings),
            "warnings": guidance_warnings,
        },
        "naming": {
            "mode_counts": dict(naming_modes),
            "lora_names": lora_names,
            "numbered_pair_count": len(numbered_records),
            "sequence_minimum": min(sequences) if sequences else None,
            "sequence_maximum": max(sequences) if sequences else None,
            "stable_sequence_complete": len(sequences) == len(numbered_records),
        },
        "distribution": {
            "orientation_counts": dict(orientation_counts),
            "resolution_counts": dict(resolution_counts),
            "crop_status_counts": dict(Counter(
                record.get("crop_status", "not_requested") for record in eligible_records
            )),
            "face_visible_items": sum(
                1 for record in eligible_records
                if int(_json_object(record.get("analysis_json")).get("counts", {}).get("faces", 0)) > 0
            ),
            "person_visible_items": sum(
                1 for record in eligible_records
                if int(_json_object(record.get("analysis_json")).get("counts", {}).get("persons", 0)) > 0
            ),
        },
        "review": {
            "failed": [
                {
                    **_image_reference(record),
                    "reason": record.get("error") or "Failed",
                }
                for record in failed_records
            ],
            "excluded": [
                {
                    **_image_reference(record),
                    "reason": record.get("error") or "Excluded",
                    "review_status": record.get("review_status", "not_requested"),
                }
                for record in excluded_records
            ],
        },
    }
