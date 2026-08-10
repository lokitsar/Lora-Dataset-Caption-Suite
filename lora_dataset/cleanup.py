import hashlib
import json
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_KLEIN_CLEANUP_PROMPT = (
    "Remove every visible watermark, logo, signature, username, URL, timestamp, stock-photo mark, "
    "caption, sticker, interface element, and overlaid text. Reconstruct only the pixels hidden by "
    "those additions using the surrounding image context. Preserve everything else exactly: subject "
    "identity, facial features, anatomy, pose, expression, clothing, objects, background, composition, "
    "crop, perspective, lighting, color, texture, sharpness, and visual style. Do not add, beautify, "
    "restyle, reframe, or alter any other content. If no removable overlay is present, keep the image "
    "visually unchanged."
)


def cleanup_config_version(config):
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class CleanupProvider(ABC):
    @abstractmethod
    def clean(self, image_path, context=None):
        raise NotImplementedError


class KleinCleanupProvider(CleanupProvider):
    def __init__(self, config):
        self.config = dict(config)
        self.model = None
        self.clip = None
        self.vae = None

    def clean(self, image_path, context=None):
        self._load_models()
        image_tensor = self._load_image_tensor(image_path)
        output = self._sample(image_tensor)
        self._save_tensor_png(output, image_path)
        return {
            "status": "cleaned_universal",
            "prompt_version": cleanup_config_version(self.config),
        }

    def _load_models(self):
        if self.model is not None:
            return
        from nodes import CLIPLoader, UNETLoader, VAELoader

        self.model = UNETLoader().load_unet(
            self.config["diffusion_model"], self.config.get("weight_dtype", "default")
        )[0]
        self.clip = CLIPLoader().load_clip(
            self.config["text_encoder"], type="flux2", device="default"
        )[0]
        self.vae = VAELoader().load_vae(self.config["vae"])[0]

    @staticmethod
    def _load_image_tensor(image_path):
        import torch

        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def _sample(self, image):
        from nodes import CLIPTextEncode, ConditioningZeroOut, VAEDecode, VAEEncode
        from comfy_extras.nodes_custom_sampler import (
            CFGGuider,
            KSamplerSelect,
            RandomNoise,
            SamplerCustomAdvanced,
        )
        from comfy_extras.nodes_edit_model import ReferenceLatent
        from comfy_extras.nodes_flux import EmptyFlux2LatentImage, Flux2Scheduler
        from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels

        scaled = ImageScaleToTotalPixels.execute(
            image,
            "lanczos",
            float(self.config.get("megapixels", 1.0)),
            16,
        )[0]
        height, width = int(scaled.shape[1]), int(scaled.shape[2])
        positive = CLIPTextEncode().encode(self.clip, self.config["prompt"])[0]
        negative = ConditioningZeroOut().zero_out(positive)[0]
        reference = VAEEncode().encode(self.vae, scaled)[0]
        positive = ReferenceLatent.execute(positive, reference)[0]
        negative = ReferenceLatent.execute(negative, reference)[0]
        latent = EmptyFlux2LatentImage.execute(width, height, 1)[0]
        noise = RandomNoise.execute(int(self.config.get("seed", 0)))[0]
        guider = CFGGuider.execute(self.model, positive, negative, 1.0)[0]
        sampler = KSamplerSelect.execute("euler")[0]
        sigmas = Flux2Scheduler.execute(int(self.config.get("steps", 4)), width, height)[0]
        samples = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]
        return VAEDecode().decode(self.vae, samples)[0]

    @staticmethod
    def _save_tensor_png(image, output_path):
        array = image[0].detach().cpu().numpy()
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        output = Path(output_path)
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        try:
            Image.fromarray(array, mode="RGB").save(temporary, format="PNG", compress_level=6)
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()


def create_cleanup_provider(config):
    return KleinCleanupProvider(config)
