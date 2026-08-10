import math

from PIL import Image


PROVIDER_MAX_MEGAPIXELS = {
    "NanoGPT": 1.0,
}


def provider_max_megapixels(provider):
    return PROVIDER_MAX_MEGAPIXELS.get(str(provider), None)


def scale_for_provider(image, provider):
    limit = provider_max_megapixels(provider)
    if limit is None:
        return image
    return scale_to_max_megapixels(image, limit)


def scale_to_max_megapixels(image, max_megapixels=1.0):
    if max_megapixels <= 0:
        raise ValueError("max_megapixels must be greater than zero")
    width, height = image.size
    maximum_pixels = int(max_megapixels * 1_000_000)
    if width * height <= maximum_pixels:
        return image

    scale = math.sqrt(maximum_pixels / (width * height))
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(target, Image.Resampling.LANCZOS)
