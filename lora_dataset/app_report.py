from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPORT_WIDTH = 1280
REPORT_MAX_LINES = 72


def _font(size, bold=False):
    candidates = (
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def _wrap_line(draw, text, font, maximum_width):
    if not text:
        return [""]
    indent = "  " if text.startswith("- ") else ""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > maximum_width:
            lines.append(current)
            current = f"{indent}{word}" if indent else word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def render_dataset_summary(summary, output_path):
    """Render the text summary as an App Mode-compatible PNG result."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body_font = _font(24)
    body_bold = _font(24, bold=True)
    title_font = _font(40, bold=True)
    measure = Image.new("RGB", (REPORT_WIDTH, 200), "#111827")
    measure_draw = ImageDraw.Draw(measure)
    rendered_lines = []
    source_lines = str(summary or "No dataset report was produced.").splitlines()
    for source_line in source_lines:
        font = body_bold if source_line in {"TRAINING READY", "ATTENTION REQUIRED", "NOT READY", "DATASET REPORT"} else body_font
        for wrapped in _wrap_line(measure_draw, source_line, font, REPORT_WIDTH - 128):
            rendered_lines.append((wrapped, font, source_line))
    if len(rendered_lines) > REPORT_MAX_LINES:
        hidden = len(rendered_lines) - REPORT_MAX_LINES
        rendered_lines = rendered_lines[:REPORT_MAX_LINES]
        rendered_lines.append((f"… {hidden} additional line(s) are available in dataset_report.json", body_font, ""))

    line_height = 38
    height = max(720, 170 + len(rendered_lines) * line_height + 72)
    image = Image.new("RGB", (REPORT_WIDTH, height), "#0b1220")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((32, 32, REPORT_WIDTH - 32, height - 32), radius=24, fill="#111827", outline="#334155", width=2)
    draw.text((72, 66), "LoRA Dataset Builder", font=title_font, fill="#f8fafc")
    draw.text((74, 120), "Automated dataset audit", font=body_font, fill="#94a3b8")
    draw.line((72, 160, REPORT_WIDTH - 72, 160), fill="#334155", width=2)

    y = 188
    for text, font, source in rendered_lines:
        if source == "TRAINING READY":
            color = "#4ade80"
        elif source in {"ATTENTION REQUIRED", "NOT READY"}:
            color = "#fb7185"
        elif source == "DATASET REPORT":
            color = "#60a5fa"
        elif text.startswith("- [FAILED]") or text.startswith("- [EXCLUDED]"):
            color = "#fbbf24"
        elif text.startswith("Full report:"):
            color = "#94a3b8"
        else:
            color = "#e2e8f0"
        draw.text((76, y), text, font=font, fill=color)
        y += line_height

    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        image.save(temporary, format="PNG", compress_level=6)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path
