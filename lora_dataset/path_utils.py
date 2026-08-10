import os
from pathlib import Path


def normalized_path(value, base=None):
    text = os.path.expandvars(os.path.expanduser(str(value).strip().strip('"')))
    if not text:
        raise ValueError("Path cannot be empty")

    path = Path(text)
    if not path.is_absolute():
        root = Path(base) if base is not None else Path.cwd()
        path = root / path
    return path.resolve(strict=False)


def path_key(path):
    return os.path.normcase(os.path.normpath(str(normalized_path(path))))


def is_within(path, parent):
    child_path = normalized_path(path)
    parent_path = normalized_path(parent)
    try:
        child_path.relative_to(parent_path)
    except ValueError:
        return False
    return True


def ensure_directory(path):
    directory = normalized_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
