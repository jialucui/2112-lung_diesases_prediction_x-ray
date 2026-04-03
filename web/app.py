# web/app.py
"""
Lung Disease Prediction API - Complete FastAPI Application
支持多任务学习：肺炎分类 + 严重程度预测
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import io
import torch
from pathlib import Path
import uvicorn
from datetime import datetime

# Import your prediction module
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
    title="肺部疾病辅助诊断系统 API",
    description="基于深度学习的胸部X光肺部疾病分类和严重程度预测",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# CORS 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型加载状态
predictor: Optional[PneumoniaPredictor] = None
model_loaded: bool = False
load_error: Optional[str] = None

# ============================================================================
# 数据模型
# ============================================================================

class PredictionResult(BaseModel):
    """预测结果数据模型"""
    image_name: str
    model_type: str
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    severity_estimated_percent: Optional[float] = None
    severity_bin_probabilities: Optional[Dict[str, float]] = None
    severity_interpretation: Optional[str] = None
    timestamp: str
    processing_time_ms: float


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    device: Optional[str] = None
    timestamp: str


class ReportRequest(BaseModel):
    """生成报告请求"""
    image_path: str
    include_details: bool = True


# ============================================================================
# 启动和关闭事件
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    global predictor, model_loaded, load_error
    
    try:
        logger.info("正在加载模型...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        
        predictor = PneumoniaPredictor.from_config_file(
            config_path='configs/config.yaml',
            checkpoint_path='checkpoints/best_model.pth',
            device=device,
            use_dataset_normalization=True
        )
        
        model_loaded = True
        logger.info("✅ 模型加载成功")
        
    except Exception as e:
        load_error = str(e)
        model_loaded = False
        logger.error(f"❌ 模型加载失败: {load_error}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global predictor
    if predictor:
        del predictor
        torch.cuda.empty_cache()
        logger.info("资源已清理")


# ============================================================================
# API 端点
# ============================================================================

@app.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check():
    """
    健康检查端点
    
    Returns:
        HealthCheckResponse: 应用状态
    """
    return HealthCheckResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """
    预测端点 - 上传X光图像进行预测
    
    Args:
        file: 上传的图像文件 (JPG, PNG等)
    
    Returns:
        PredictionResult: 预测结果
        
    Raises:
        HTTPException: 模型未加载或预测失败
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"模型未加载: {load_error}"
        )
    
    try:
        import time
        start_time = time.time()
        
        # 验证文件类型
        if file.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file.content_type}"
            )
        
        # 读取文件内容
        contents = await file.read()
        temp_image_path = f"/tmp/{file.filename}"
        
        with open(temp_image_path, 'wb') as f:
            f.write(contents)
        
        # 进行预测
        logger.info(f"预测图像: {file.filename}")
        prediction_result = predictor.predict(temp_image_path)
        
        processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 构建响应
        result = PredictionResult(
            image_name=file.filename,
            model_type=prediction_result.get('model_type'),
            predicted_class=prediction_result.get('predicted_class'),
            confidence=prediction_result.get('class_probabilities', {}).get(
                prediction_result.get('predicted_class'), 0
            ),
            class_probabilities=prediction_result.get('class_probabilities', {}),
            severity_estimated_percent=prediction_result.get('severity_estimated_percent'),
            severity_bin_probabilities=prediction_result.get('severity_bin_probabilities'),
            severity_interpretation=prediction_result.get('severity_interpretation'),
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
        logger.info(f"预测完成: {result.predicted_class} (置信度: {result.confidence:.2%})")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"预测失败: {str(e)}"
        )
    finally:
        # 清理临时文件
        import os
        try:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        except:
            pass


@app.post("/api/v1/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    批量预测端点
    
    Args:
        files: 上传的多个图像文件
    
    Returns:
        list: 预测结果列表
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"模型未加载: {load_error}"
        )
    
    results = []
    for file in files:
        try:
            # 临时调用单个预测端点
            temp_result = await predict(file)
            results.append(temp_result)
        except HTTPException as e:
            results.append({
                "image_name": file.filename,
                "error": e.detail
            })
    
    return {
        "total": len(files),
        "successful": sum(1 for r in results if 'error' not in r),
        "results": results
    }


@app.get("/api/v1/report/{image_name}")
async def get_report(image_name: str, include_details: bool = True):
    """
    获取预测报告
    
    Args:
        image_name: 图像文件名
        include_details: 是否包含详细信息
    
    Returns:
        dict: 格式化的诊断报告
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="模型未加载"
        )
    
    try:
        # 这里可以从缓存或数据库中获取之前的预测结果
        # 或重新预测
        prediction_result = predictor.predict(image_name)
        
        report = PneumoniaPredictor.format_report(prediction_result)
        
        return {
            "image_name": image_name,
            "report": report,
            "raw_prediction": prediction_result if include_details else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"图像未找到: {image_name}"
        )
    except Exception as e:
        logger.error(f"生成报告失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"生成报告失败: {str(e)}"
        )


@app.get("/api/v1/info")
async def get_info():
    """获取API和模型信息"""
    return {
        "app_name": "肺部疾病辅助诊断系统",
        "version": "1.0.0",
        "api_version": "v1",
        "model_info": {
            "loaded": model_loaded,
            "type": predictor.model_type if predictor else None,
            "classes": predictor.class_names if predictor else None,
            "device": 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        "endpoints": {
            "predict": "/api/v1/predict (POST)",
            "predict_batch": "/api/v1/predict_batch (POST)",
            "health": "/api/v1/health (GET)",
            "info": "/api/v1/info (GET)",
            "docs": "/api/v1/docs (GET)"
        }
    }


@app.get("/")
async def root():
    """根路由 - 重定向到文档"""
    return {
        "message": "欢迎使用肺部疾病辅助诊断系统 API",
        "docs": "/api/v1/docs",
        "health": "/api/v1/health",
        "info": "/api/v1/info"
    }


# ============================================================================
# 错误处理
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    logger.error(f"HTTP异常: {exc.status_code} - {exc.detail}")
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
    """通用异常处理"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "服务器内部错误",
            "timestamp": datetime.now().isoformat()
        }
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