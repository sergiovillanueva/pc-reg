"""Frozen DINOv3 features: patch tokens and head-averaged CLS-to-patch attention of layer -6.

Two passes, as in the paper:
  1. patch tokens: SDPA attention, fp16 autocast, hidden_states[-6] without the CLS and register tokens -> float32
  2. CLS attention: eager attention in fp32, attentions[-6] averaged over heads, CLS row over the patches -> float16
Images are resized to 448 x 448 (ViT-L/16, 28 x 28 grid) or 512 x 512 (ViT-H+/16, 32 x 32 grid), ImageNet normalization.
"""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

from .config import BACKBONES, LAYER_IDX


class FeatureExtractor:
    def __init__(self, backbone: str = "vitL", device: str | None = None):
        self.cfg = BACKBONES[backbone]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_patches = self.cfg["grid"] ** 2
        r = self.cfg["resolution"]
        self.tf = transforms.Compose([transforms.Resize((r, r)), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self._feat_model = None
        self._attn_model = None

    def _batch(self, paths):
        return torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths]).to(self.device)

    def _start(self, model):
        return 1 + getattr(model.config, "num_register_tokens", 4)  # skip CLS + register tokens

    @torch.no_grad()
    def patch_tokens(self, paths: list[str], batch_size: int = 8) -> np.ndarray:
        if self._feat_model is None:
            self._feat_model = AutoModel.from_pretrained(self.cfg["hf_name"]).to(self.device).eval()
        m, out = self._feat_model, []
        s = self._start(m)
        for i in range(0, len(paths), batch_size):
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                o = m(self._batch(paths[i:i + batch_size]), output_hidden_states=True)
            out.append(o.hidden_states[LAYER_IDX][:, s:s + self.n_patches, :].float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def cls_attention(self, paths: list[str], batch_size: int = 4) -> np.ndarray:
        if self._attn_model is None:
            self._attn_model = AutoModel.from_pretrained(self.cfg["hf_name"], attn_implementation="eager").to(self.device).eval()
        m, out = self._attn_model, []
        s = self._start(m)
        for i in range(0, len(paths), batch_size):
            o = m(self._batch(paths[i:i + batch_size]), output_attentions=True)
            attn = o.attentions[LAYER_IDX].float().mean(dim=1)  # (B, seq, seq), averaged over heads
            out.append(attn[:, 0, s:s + self.n_patches].cpu().numpy())
        return np.concatenate(out).astype(np.float16)
