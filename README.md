# LoRA Dataset Caption Suite

An automated, resumable image and video LoRA dataset builder for ComfyUI. Image mode
creates exact PNG/TXT training pairs, removes visible overlays with Flux 2 Klein 9B,
verifies cleanup fidelity, analyzes and crops images, and generates model-specific
positive captions. Video mode uses FFmpeg to create consistent MP4/TXT pairs and
captions temporal content from ordered frames sampled across each clip.

Supported training recipes are **Krea 2**, **Anima**, and **MiniMax H3**, with
separate caption guidance for Character, Style, and Concept LoRAs.

## Highlights

- Queue once to process the complete eligible set; there is no batch-count control.
- Resume processes only new or changed sources. Failed items can be retried, and a
  revisioned force rebuild intentionally regenerates the full active set.
- JPEG, AVIF, WebP, and other Pillow-readable sources become PNG before dataset output.
- MP4, MOV, MKV, WebM, AVI, MPEG, M4V, and FLV sources can be trimmed, resampled,
  resized, cropped or padded, and encoded as H.264 MP4 dataset clips.
- Video defaults target memory-stable H3 clips: 24 fps, exactly 107 frames,
  landscape 896x512 or portrait 512x896.
  Duration, start time, frame rate, dimensions, resize/crop strategy, padding,
  audio retention, CRF, and encoder preset are configurable.
- `keep_native` preserves every source frame's original dimensions and aspect ratio;
  width, height, and crop position are ignored. If a dimension is odd, FFmpeg pads
  only the right or bottom edge by one pixel for H.264/yuv420p compatibility rather
  than scaling or cropping visible content. AI-toolkit can then bucket/resize later.
- Video captioning sends one ordered set of evenly distributed frames to the selected
  vision API so the result describes appearance, motion, action progression, camera
  behavior, and ending state rather than one isolated frame.
- Video captions use a dedicated MiniMax H3 LoRA policy: natural English, visible
  evidence only, explicit subject-versus-camera motion, useful secondary motion,
  chronological progression, no uncertainty language, and no generic quality tags.
- Video mode requires the dedicated MiniMax H3 profile. Krea 2 and Anima remain
  image profiles and are rejected in video mode rather than silently producing
  unsuitable sidecars.
- Originals are never modified.
- Optional Flux 2 Klein 9B cleanup removes watermarks, logos, signatures, URLs,
  timestamps, overlay text, and similar artifacts before captioning.
- Cleanup verification always checks excessive visual changes. Residual watermark
  scanning is optional; the default trusts Klein so legitimate clothing and scene
  text are not rejected.
- Ultralytics subject and face detection supports identity-preserving crops.
- Caption providers include Ollama, OpenRouter, NanoGPT, and Kobold-compatible APIs.
- Provider images sent to NanoGPT are capped at one megapixel without shrinking the
  final training image.
- Positive captions describe visible content only. Negative prompts, absence notes,
  editing instructions, and caption-model thought notes are filtered out.
- Exact duplicates are excluded automatically; near duplicates and quality warnings
  are reported for review.
- Krea 2 reports include the Raw-model/1024-bucket handoff, native source-area
  coverage, missing-trigger guidance, and recurring caption descriptors that may
  contribute to trigger bleed.
- Optional stable naming creates exact pairs such as `taarna_0001.png` and
  `taarna_0001.txt` without renumbering existing items.
- A SQLite manifest makes every stage resumable and auditable.
- Official ComfyUI App Mode exposes the complete workflow through a focused UI with
  live progress and a visual training-readiness report.

## Installation

Clone into ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lokitsar/Lora-Dataset-Caption-Suite.git
pip install -r Lora-Dataset-Caption-Suite/requirements.txt
```

Use the Python environment that launches ComfyUI. Restart ComfyUI and hard-refresh
the browser after installation.

If you previously used the development copy embedded in
`ComfyUI-Lokitsars-Nodes`, do not load both implementations at the same time because
they expose the same LoRA dataset node IDs.

## App Mode

Open **Templates** in ComfyUI's left sidebar, select the **App** tab, and search for
`LoRA Dataset Builder`. Choose **LoRA Dataset Builder.app**.

The App promotes these controls while keeping dataset logic in the Python backend:

- source and destination directories
- image or video media mode
- Krea 2, Anima, or MiniMax H3 recipe, LoRA type, trigger, and additional instructions
- caption provider, API URL/key, API model discovery, and model selection
- installed Klein diffusion model, text encoder, and VAE
- watermark verification mode, watermark model, subject model, and face model
- resume/retry/rebuild behavior
- preserved or stable numbered output naming
- FFmpeg trim, exact fps/frame count, orientation-specific size, crop/pad,
  orientation filtering, encoding, audio, and caption-frame controls

Press **Run** once. The backend automatically processes the full set selected by the
run mode. On completion, App Mode displays the visual Dataset Report.

## Processing order

```text
discover source images
  -> normalize working output to PNG
  -> Klein universal overlay cleanup
  -> image-fidelity verification (optional residual-artifact scan)
  -> subject and face analysis
  -> identity-preserving crop
  -> model/type-specific positive caption
  -> exact PNG/TXT validation
  -> duplicate, quality, distribution, and readiness report
```

Video mode uses a separate media path:

```text
discover source videos
  -> FFmpeg trim and duration normalization
  -> exact fps and optional exact-frame conversion (short clips clone-pad)
  -> one landscape target and one portrait target, or a legacy single target
  -> optional landscape-only or portrait-only filtering
  -> keep native size, crop-to-fill, fit-within, pad-to-fit, or stretch
  -> H.264 MP4 encoding (optional AAC audio)
  -> ordered frame sampling across the prepared clip
  -> one temporal positive caption through the configured vision API
  -> exact MP4/TXT validation
  -> duplicate, resolution, duration, caption, and readiness report
```

Image cleanup, still-image detection, and smart crop providers are intentionally
skipped in video mode; FFmpeg owns spatial preparation there.

For a stable MiniMax H3 training dataset, use `normalize_by_orientation` with
`crop_to_fill` or `pad_to_fit`, choose one landscape and one portrait size, set
24 FPS, and set an exact frame count such as 107. `fit_within` is rejected with
orientation normalization because it would retain variable output dimensions and
reintroduce heterogeneous AI Toolkit buckets. After preprocessing, configure AI
Toolkit with one training resolution family and disable automatic frame-count
selection if you want to preserve the normalized temporal bucket.

Confirmed bad cleanup results are excluded so the remaining eligible set can still
become training-ready. Provider, detector, or system failures are blocking errors;
they are not treated as proof that an image is bad.

## Output structure

The destination directory contains:

```text
dataset/                  final PNG/TXT or MP4/TXT pairs
manifest/dataset.db       resumable SQLite state
reports/dataset_report.json
review/                   deterministic evidence and excluded/review items
```

The Run Summary reports failed and excluded filenames with their recorded reasons,
plus duplicates, image quality, caption length, orientation, crop outcomes, subject
visibility, naming state, Krea 2 training targets, 1024 source coverage, and
recurring-caption guidance.

## Included nodes

- LoRA Dataset Source
- LoRA Dataset Profile
- LoRA Dataset Caption Provider
- LoRA Dataset Klein 9B Cleanup
- LoRA Dataset Cleanup Verifier
- LoRA Dataset Image Analyzer
- LoRA Dataset Smart Crop
- LoRA Dataset Video Prep (FFmpeg)
- LoRA Dataset Builder
- LoRA Dataset Run Summary
- LoRA Dataset App Report
- LoRA Dataset Validator

## Development

Run the regression suite from the repository root:

```bash
python -m pip install aiohttp Pillow numpy pytest
python -m pytest -q
```

The test suite covers manifest/resume behavior, PNG conversion, caption policies,
cleanup verification, analysis/cropping, video preparation and temporal sampling,
dataset intelligence, stable naming, live progress, and App Mode workflow configuration.

## License

MIT
