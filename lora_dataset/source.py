import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .path_utils import is_within, normalized_path


BASE_IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def supported_image_extensions():
    extensions = set(BASE_IMAGE_EXTENSIONS)
    registered = {suffix.lower() for suffix in Image.registered_extensions()}
    if ".avif" in registered:
        extensions.add(".avif")
    return extensions


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceItem:
    item_id: str
    path: Path
    relative_path: str
    content_hash: str
    size: int
    mtime_ns: int


class DatasetSource:
    def __init__(self, source_directory, recursive=True, excluded_directories=None):
        self.root = normalized_path(source_directory)
        if not self.root.is_dir():
            raise NotADirectoryError(f"Source directory does not exist: {self.root}")
        self.recursive = bool(recursive)
        self.excluded_directories = [normalized_path(path) for path in (excluded_directories or [])]

    def discover(self):
        iterator = self.root.rglob("*") if self.recursive else self.root.glob("*")
        extensions = supported_image_extensions()
        paths = []
        for path in iterator:
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if any(is_within(path, excluded) for excluded in self.excluded_directories):
                continue
            paths.append(path.resolve(strict=False))

        items = []
        for path in sorted(paths, key=lambda value: value.relative_to(self.root).as_posix().casefold()):
            relative_path = path.relative_to(self.root).as_posix()
            identity = relative_path.casefold().encode("utf-8")
            stat = path.stat()
            items.append(SourceItem(
                item_id=hashlib.sha256(identity).hexdigest()[:24],
                path=path,
                relative_path=relative_path,
                content_hash=sha256_file(path),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            ))
        return items
