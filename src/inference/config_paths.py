"""Default config and checkpoint paths for inference (web + CLI)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_CANDIDATES = (
    "src/configs/config_chest_xray.yaml",
    "src/configs/config.yaml",
)

DEFAULT_CHEST_CHECKPOINT = "checkpoints/chest_xray_stopped_epoch7/best_model.pth"


def resolve_config_path(
    project_root: Optional[Path] = None,
    *,
    env_var: str = "LUNG_XRAY_CONFIG",
) -> Path:
    root = project_root or PROJECT_ROOT
    env_cfg = os.environ.get(env_var)
    if env_cfg:
        p = Path(env_cfg)
        if not p.is_absolute():
            p = root / env_cfg
        if p.is_file():
            return p
    for rel in CONFIG_CANDIDATES:
        p = root / rel
        if p.is_file():
            return p
    return root / CONFIG_CANDIDATES[0]


def default_checkpoint_path(
    project_root: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    env_var: str = "LUNG_XRAY_CHECKPOINT",
    fallback: str = DEFAULT_CHEST_CHECKPOINT,
) -> Path:
    root = project_root or PROJECT_ROOT
    env_ck = os.environ.get(env_var)
    if env_ck:
        p = Path(env_ck)
        return p if p.is_absolute() else root / p
    if config:
        ck_dir = (config.get("paths") or {}).get("checkpoint_dir")
        if ck_dir:
            base = root / ck_dir if not Path(ck_dir).is_absolute() else Path(ck_dir)
            candidate = base / "best_model.pth"
            if candidate.is_file():
                return candidate
    return root / fallback
