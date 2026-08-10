# LoRA Dataset Caption Suite

An automated, resumable LoRA dataset builder for ComfyUI. It converts supported
source images to exact PNG/TXT training pairs, removes visible overlays with Flux 2
Klein 9B, verifies cleanup fidelity, analyzes and crops images, generates
model-specific positive captions, and audits the finished dataset.

The first supported training recipes are **Krea 2** and **Anima**, with separate
caption guidance for Character, Style, and Concept LoRAs.

## Highlights

- Queue once to process the complete eligible set; there is no batch-count control.
- Resume processes only new or changed sources. Failed items can be retried, and a
  revisioned force rebuild intentionally regenerates the full active set.
- JPEG, AVIF, WebP, and other Pillow-readable sources become PNG before dataset output.
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
- Krea 2 or Anima recipe, LoRA type, trigger, and additional caption instructions
- caption provider, API URL/key, API model discovery, and model selection
- installed Klein diffusion model, text encoder, and VAE
- watermark verification mode, watermark model, subject model, and face model
- resume/retry/rebuild behavior
- preserved or stable numbered output naming

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

Confirmed bad cleanup results are excluded so the remaining eligible set can still
become training-ready. Provider, detector, or system failures are blocking errors;
they are not treated as proof that an image is bad.

## Output structure

The destination directory contains:

```text
dataset/                  final PNG/TXT pairs
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
cleanup verification, analysis/cropping, dataset intelligence, stable naming, live
progress, and App Mode workflow configuration.

## License

MIT
