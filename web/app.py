# web/app.py
"""
Lung Disease Prediction API — chest X-ray classification + severity (assistive).
Home page: upload an image and view probabilities + severity.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _resolve_config_yaml() -> str:
    """Config path: env LUNG_XRAY_CONFIG, else chest deploy config, else default."""
    env_cfg = os.environ.get("LUNG_XRAY_CONFIG")
    if env_cfg:
        p = Path(env_cfg)
        if not p.is_absolute():
            p = _REPO_ROOT / env_cfg
        if p.is_file():
            return str(p)
    for rel in (
        "src/configs/config_chest_xray.yaml",
        "src/configs/config.yaml",
        "configs/config.yaml",
    ):
        p = _REPO_ROOT / rel
        if p.is_file():
            return str(p)
    return str(_REPO_ROOT / "src/configs/config_chest_xray.yaml")


from src.inference.predictor import PneumoniaPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 初始化应用和模型
# ============================================================================
app = FastAPI(
    title="Lung X-ray assistive API",
    description="Chest X-ray class probabilities and severity estimate (not medical advice)",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Optional[PneumoniaPredictor] = None
model_loaded: bool = False
load_error: Optional[str] = None

# ============================================================================
# 数据模型
# ============================================================================

class PredictionResult(BaseModel):
    """Prediction payload returned by /api/v1/predict."""

    image_name: str
    model_type: str
    predicted_class: str
    confidence: float
    confidence_score: Optional[float] = None
    confidence_threshold: Optional[float] = None
    triage_message: Optional[str] = None
    class_probabilities: Dict[str, float]
    pneumonia_probability: Optional[float] = None
    subtype: Optional[Dict[str, float]] = None
    severity_estimated_percent: Optional[float] = None
    severity_bin_probabilities: Optional[Dict[str, float]] = None
    severity_interpretation: Optional[str] = None
    severity_note: Optional[str] = None
    needs_manual_review: Optional[bool] = None
    report: Optional[str] = None
    grad_cam_image: Optional[str] = None
    grad_cam_target_class: Optional[str] = None
    timestamp: str
    processing_time_ms: float


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: Optional[str] = None
    timestamp: str


# ============================================================================
# 启动和关闭事件
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor, model_loaded, load_error

    try:
        logger.info("Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Device: %s", device)

        ckpt = os.environ.get(
            "LUNG_XRAY_CHECKPOINT",
            "checkpoints/chest_xray_stopped_epoch7/best_model.pth",
        )
        # LUNG_XRAY_NO_DATASET_NORM=1 forces ImageNet stats (overconfident on this model).
        use_ds_norm = os.environ.get("LUNG_XRAY_NO_DATASET_NORM", "").lower() not in (
            "1",
            "true",
            "yes",
        )
        predictor = PneumoniaPredictor.from_config_file(
            config_path=_resolve_config_yaml(),
            checkpoint_path=ckpt,
            device=device,
            use_dataset_normalization=use_ds_norm,
        )

        model_loaded = True
        logger.info("Model loaded successfully")

    except Exception as e:
        load_error = str(e)
        model_loaded = False
        logger.error("Model load failed: %s", load_error)


@app.on_event("shutdown")
async def shutdown_event():
    global predictor
    if predictor:
        del predictor
        torch.cuda.empty_cache()
        logger.info("Shutdown cleanup done")


# ============================================================================
# API 端点
# ============================================================================

@app.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        timestamp=datetime.now().isoformat()
    )


def _allowed_image(file: UploadFile) -> bool:
    ct = (file.content_type or "").lower()
    if ct in ("image/jpeg", "image/png", "image/jpg"):
        return True
    name = (file.filename or "").lower()
    return name.endswith((".jpg", ".jpeg", ".png"))


@app.post("/api/v1/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    age: Optional[float] = Form(None),
    gender: Optional[str] = Form(None),
):
    """Upload a chest image; optional age/gender used when model.tabular_dim > 0."""
    if not model_loaded or predictor is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    temp_image_path: Optional[str] = None
    try:
        start_time = time.time()

        if not _allowed_image(file):
            raise HTTPException(
                status_code=400,
                detail=f"Please upload JPG or PNG (Content-Type was: {file.content_type})",
            )

        contents = await file.read()
        suffix = Path(file.filename or "upload.jpg").suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png"):
            suffix = ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            temp_image_path = tmp.name

        logger.info("Predict image: %s", file.filename)
        inf_cfg = predictor.config.get("inference") or {}
        thr = float(inf_cfg.get("confidence_threshold", 0.8))
        triage_msg = inf_cfg.get("triage_message", "Low confidence — manual review recommended.")

        prediction_result = predictor.predict(
            temp_image_path, age=age, gender=gender, confidence_threshold=thr
        )
        prediction_en = PneumoniaPredictor.result_for_english_ui(prediction_result)

        processing_time = (time.time() - start_time) * 1000
        probs = prediction_en.get("class_probabilities") or {}
        pred_class = prediction_en.get("predicted_class") or ""
        confidence = float(
            prediction_en.get("confidence_score") or probs.get(pred_class, 0.0)
        )

        report_text = PneumoniaPredictor.format_report(prediction_en, language="en")

        grad_cam_image = None
        grad_cam_target_class = None
        try:
            gc = predictor.grad_cam(
                temp_image_path,
                target_class_index=prediction_en.get("predicted_class_index"),
                age=age,
                gender=gender,
            )
            grad_cam_image = gc.get("grad_cam_image")
            grad_cam_target_class = gc.get("target_class")
        except Exception as gc_err:
            logger.warning("Grad-CAM failed (prediction still returned): %s", gc_err)

        sub_raw = prediction_en.get("subtype") or {}
        subtype_out = {k: float(v) for k, v in sub_raw.items()} if sub_raw else None
        result = PredictionResult(
            image_name=file.filename or "upload",
            model_type=prediction_en.get("model_type", ""),
            predicted_class=pred_class,
            confidence=confidence,
            confidence_score=confidence,
            confidence_threshold=thr,
            triage_message=triage_msg if prediction_en.get("needs_manual_review") else None,
            class_probabilities={k: float(v) for k, v in probs.items()},
            pneumonia_probability=prediction_en.get("pneumonia_probability"),
            subtype=subtype_out,
            severity_estimated_percent=prediction_en.get("severity_estimated_percent"),
            severity_bin_probabilities=prediction_en.get("severity_bin_probabilities"),
            severity_interpretation=prediction_en.get("severity_interpretation"),
            severity_note=prediction_en.get("severity_note"),
            needs_manual_review=prediction_en.get("needs_manual_review"),
            report=report_text,
            grad_cam_image=grad_cam_image,
            grad_cam_target_class=grad_cam_target_class,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
        )

        logger.info("Prediction done: %s (confidence %.2f%%)", result.predicted_class, result.confidence * 100)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from e
    finally:
        if temp_image_path and os.path.isfile(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except OSError:
                pass


@app.get("/api/v1/info")
async def get_info():
    """API and model metadata."""
    return {
        "app_name": "Lung X-ray assistive API",
        "version": "1.0.0",
        "api_version": "v1",
        "model_info": {
            "loaded": model_loaded,
            "type": predictor.model_type if predictor else None,
            "classes": predictor.class_names if predictor else None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "endpoints": {
            "web_ui": "/ (GET) — browser upload UI",
            "predict": "/api/v1/predict (POST, multipart: file, optional age, gender)",
            "health": "/api/v1/health (GET)",
            "info": "/api/v1/info (GET)",
            "docs": "/api/v1/docs (GET)",
        },
    }


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
async def root():
    """Web UI: upload → class probabilities + severity."""
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index, media_type="text/html; charset=utf-8")
    return JSONResponse(
        {
            "message": "Missing web/static/index.html",
            "docs": "/api/v1/docs",
            "health": "/api/v1/health",
        }
    )


# ============================================================================
# 错误处理
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error("HTTP %s - %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Unhandled error: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat(),
        },
    )


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )