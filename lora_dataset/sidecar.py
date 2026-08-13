import os
import re
import uuid
from pathlib import Path

from .path_utils import ensure_directory
from .video import VIDEO_EXTENSIONS


IMAGE_EXTENSIONS = {
    ".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}
_WINDOWS_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_FILENAME = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", re.IGNORECASE)


def sidecar_stem(filename):
    # Source manifests can contain paths written on a different operating
    # system. Normalize both separator styles before asking the host OS to
    # interpret the basename.
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    path = Path(name)
    if path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        name = path.stem

    name = _WINDOWS_ILLEGAL_FILENAME_CHARS.sub("_", name).rstrip(" .")
    if not name:
        raise ValueError("filename must contain at least one valid character")
    if _WINDOWS_RESERVED_FILENAME.match(name):
        name = f"_{name}"
    return name


class DatasetSidecarWriter:
    def write(self, text, filename, folder, existing_file="overwrite"):
        if existing_file not in {"overwrite", "skip"}:
            raise ValueError("existing_file must be 'overwrite' or 'skip'")

        destination = ensure_directory(folder)
        sidecar = destination / f"{sidecar_stem(filename)}.txt"
        if existing_file == "skip" and sidecar.exists():
            return {"status": "skipped", "path": str(sidecar)}

        temporary = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(str(text), encoding="utf-8", newline="")
            os.replace(temporary, sidecar)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {"status": "written", "path": str(sidecar)}
