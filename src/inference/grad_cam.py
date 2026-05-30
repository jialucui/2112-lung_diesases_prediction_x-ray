"""
Grad-CAM for chest X-ray classifiers (DenseNet multi-task / binary).
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def _resolve_target_layer(model: nn.Module, model_type: str) -> nn.Module:
    if model_type == "multi_task":
        return model.backbone.features[-1]
    if model_type == "binary":
        inner = model.model if hasattr(model, "model") else model
        if hasattr(inner, "features"):
            return inner.features[-1]
        if hasattr(inner, "layer4"):
            return inner.layer4[-1]
    raise ValueError(f"Grad-CAM not supported for model_type={model_type!r}")


def _forward_class_logits(
    model: nn.Module,
    model_type: str,
    input_tensor: torch.Tensor,
    tabular: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass without inplace ReLU (required for Grad-CAM on DenseNet)."""
    if model_type == "multi_task":
        feat = model.backbone.features(input_tensor)
        feat = F.relu(feat, inplace=False)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        if model.tabular_proj is not None:
            if tabular is None:
                tabular = torch.zeros(
                    input_tensor.shape[0],
                    model.tabular_dim,
                    device=input_tensor.device,
                    dtype=pooled.dtype,
                )
            pooled = torch.cat([pooled, model.tabular_proj(tabular)], dim=1)
        return model.binary_head(pooled)

    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "features"):
        feat = inner.features(input_tensor)
        feat = F.relu(feat, inplace=False)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return inner.classifier(pooled)

    x = inner.conv1(input_tensor)
    x = inner.bn1(x)
    x = F.relu(x, inplace=False)
    x = inner.maxpool(x)
    x = inner.layer1(x)
    x = inner.layer2(x)
    x = inner.layer3(x)
    x = inner.layer4(x)
    x = inner.avgpool(x)
    x = torch.flatten(x, 1)
    return inner.fc(x)


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0.0)
    if cam.max() > 1e-8:
        cam = cam / cam.max()
    return cam.astype(np.float32)


def compute_grad_cam(
    model: nn.Module,
    model_type: str,
    input_tensor: torch.Tensor,
    target_class: int,
    tabular: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Return Grad-CAM heatmap in [0, 1], shape (H, W) matching input spatial size.
    """
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    target_layer = _resolve_target_layer(model, model_type)

    def _fwd_hook(_module, _inp, out):
        activations["value"] = out.detach().clone()

    def _bwd_hook(_module, _inp, grad_out):
        gradients["value"] = grad_out[0].detach().clone()

    fwd_handle = target_layer.register_forward_hook(_fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(_bwd_hook)

    model.eval()
    try:
        input_tensor = input_tensor.clone().detach().requires_grad_(False)
        model.zero_grad(set_to_none=True)

        class_logits = _forward_class_logits(model, model_type, input_tensor, tabular)

        if target_class < 0 or target_class >= class_logits.shape[1]:
            raise ValueError(f"target_class {target_class} out of range [0, {class_logits.shape[1]})")

        score = class_logits[0, target_class]
        score.backward()

        acts = activations.get("value")
        grads = gradients.get("value")
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze(0)
        cam = F.relu(cam).detach().cpu().numpy()
        cam = _normalize_cam(cam)

        h, w = int(input_tensor.shape[2]), int(input_tensor.shape[3])
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def overlay_grad_cam(
    rgb_uint8: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend JET colormap heatmap onto RGB image (uint8)."""
    h, w = rgb_uint8.shape[:2]
    cam_u8 = np.uint8(255 * _normalize_cam(cam))
    heatmap_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    if heatmap_rgb.shape[:2] != (h, w):
        heatmap_rgb = cv2.resize(heatmap_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    blended = (1.0 - alpha) * rgb_uint8.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def rgb_to_base64_png(rgb_uint8: np.ndarray) -> str:
    img = Image.fromarray(rgb_uint8)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def grad_cam_visualization(
    model: nn.Module,
    model_type: str,
    input_tensor: torch.Tensor,
    rgb_uint8: np.ndarray,
    target_class: int,
    tabular: Optional[torch.Tensor] = None,
    alpha: float = 0.45,
) -> Tuple[np.ndarray, str]:
    """Return (overlay_rgb, base64_png_data_url)."""
    cam = compute_grad_cam(model, model_type, input_tensor, target_class, tabular=tabular)
    overlay = overlay_grad_cam(rgb_uint8, cam, alpha=alpha)
    return overlay, rgb_to_base64_png(overlay)
