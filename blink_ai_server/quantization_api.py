"""
量化功能API接口
提供模型量化和移动端推理的REST API
"""

import os
import json
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import base64
from io import BytesIO

from quantization import ModelConverter, ModelQuantizer
from quantization.utils import ModelUtils, PerformanceUtils
from mobile_inference import MobileFaceService

# 创建路由器
router = APIRouter(prefix="/quantization", tags=["量化功能"])

# 全局变量
model_converter = None
model_quantizer = None
mobile_face_service = None

def init_quantization_services():
    """初始化量化服务"""
    global model_converter, model_quantizer
    
    try:
        model_converter = ModelConverter()
        model_quantizer = ModelQuantizer()
        logging.info("量化服务初始化成功")
    except Exception as e:
        logging.error(f"量化服务初始化失败: {e}")

# 初始化服务
init_quantization_services()

@router.post("/convert_model")
async def convert_insightface_model(
    model_name: str = Form("buffalo_l"),
    input_width: int = Form(640),
    input_height: int = Form(640)
):
    """
    转换InsightFace模型为ONNX格式
    
    Args:
        model_name: InsightFace模型名称
        input_width: 输入图像宽度
        input_height: 输入图像高度
    
    Returns:
        Dict: 转换结果
    """
    try:
        if model_converter is None:
            raise HTTPException(status_code=500, detail="模型转换器未初始化")
        
        result = model_converter.convert_insightface_to_onnx(
            model_name=model_name,
            input_size=(input_width, input_height)
        )
        
        if result.get("success", False):
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "模型转换失败"))
            
    except Exception as e:
        logging.error(f"模型转换失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quantize_model")
async def quantize_model(
    model_path: str = Form(...),
    quantization_type: str = Form("int8"),
    calibration_images: Optional[List[UploadFile]] = File(None)
):
    """
    量化模型
    
    Args:
        model_path: 模型文件路径
        quantization_type: 量化类型 (int8, fp16, dynamic_int8)
        calibration_images: 校准图像文件列表
    
    Returns:
        Dict: 量化结果
    """
    try:
        if model_quantizer is None:
            raise HTTPException(status_code=500, detail="模型量化器未初始化")
        
        # 处理校准图像
        calibration_dataset_path = None
        if calibration_images and quantization_type == "int8":
            calibration_dataset_path = "quantization/datasets/calibration_images"
            os.makedirs(calibration_dataset_path, exist_ok=True)
            
            # 保存上传的校准图像
            for i, image_file in enumerate(calibration_images):
                image_path = os.path.join(calibration_dataset_path, f"calibration_{i}.jpg")
                with open(image_path, "wb") as buffer:
                    content = await image_file.read()
                    buffer.write(content)
        
        # 执行量化
        if quantization_type == "int8":
            if calibration_dataset_path is None:
                raise HTTPException(status_code=400, detail="INT8量化需要校准图像")
            result = model_quantizer.quantize_to_int8(model_path, calibration_dataset_path)
        elif quantization_type == "fp16":
            result = model_quantizer.quantize_to_fp16(model_path)
        elif quantization_type == "dynamic_int8":
            result = model_quantizer.quantize_dynamic_int8(model_path)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的量化类型: {quantization_type}")
        
        if result.get("success", False):
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "模型量化失败"))
            
    except Exception as e:
        logging.error(f"模型量化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quantization_status/{model_id}")
async def get_quantization_status(model_id: str):
    """
    获取量化状态
    
    Args:
        model_id: 模型ID
    
    Returns:
        Dict: 量化状态
    """
    try:
        # 这里可以实现状态查询逻辑
        # 暂时返回模拟数据
        return JSONResponse(content={
            "model_id": model_id,
            "status": "completed",
            "progress": 100,
            "message": "量化完成"
        })
        
    except Exception as e:
        logging.error(f"获取量化状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quantized_models")
async def get_quantized_models():
    """
    获取量化模型列表
    
    Returns:
        List: 量化模型列表
    """
    try:
        models_dir = "quantization/mobile_models"
        if not os.path.exists(models_dir):
            return JSONResponse(content=[])
        
        models = []
        for filename in os.listdir(models_dir):
            if filename.endswith('.onnx'):
                model_path = os.path.join(models_dir, filename)
                model_info = {
                    "filename": filename,
                    "path": model_path,
                    "size_mb": os.path.getsize(model_path) / (1024 * 1024),
                    "quantization_type": "unknown"
                }
                
                # 根据文件名判断量化类型
                if "_int8" in filename:
                    model_info["quantization_type"] = "int8"
                elif "_fp16" in filename:
                    model_info["quantization_type"] = "fp16"
                elif "_dynamic_int8" in filename:
                    model_info["quantization_type"] = "dynamic_int8"
                
                models.append(model_info)
        
        return JSONResponse(content=models)
        
    except Exception as e:
        logging.error(f"获取量化模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/mobile/detect_faces")
async def mobile_detect_faces(
    image: UploadFile = File(...),
    model_type: str = Form("int8"),
    confidence_threshold: float = Form(0.5)
):
    """
    移动端人脸检测
    
    Args:
        image: 图像文件
        model_type: 模型类型 (int8, fp16)
        confidence_threshold: 置信度阈值
    
    Returns:
        Dict: 检测结果
    """
    try:
        # 查找对应的量化模型
        models_dir = "quantization/mobile_models"
        detection_model_path = None
        
        for filename in os.listdir(models_dir):
            if "face_detection" in filename and f"_{model_type}" in filename:
                detection_model_path = os.path.join(models_dir, filename)
                break
        
        if detection_model_path is None:
            raise HTTPException(status_code=404, detail=f"未找到 {model_type} 类型的检测模型")
        
        # 初始化移动端服务
        mobile_service = MobileFaceService(
            detection_model_path=detection_model_path,
            recognition_model_path=detection_model_path  # 临时使用检测模型
        )
        
        # 保存上传的图像
        image_data = await image.read()
        
        # 执行人脸检测
        faces = mobile_service.detect_faces(image_data, confidence_threshold)
        
        return JSONResponse(content={
            "message": "人脸检测完成",
            "model_type": model_type,
            "face_count": len(faces),
            "faces": faces
        })
        
    except Exception as e:
        logging.error(f"移动端人脸检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/mobile/recognize_faces")
async def mobile_recognize_faces(
    image: UploadFile = File(...),
    model_type: str = Form("int8"),
    similarity_threshold: float = Form(0.6),
    known_embeddings: str = Form("[]")
):
    """
    移动端人脸识别
    
    Args:
        image: 图像文件
        model_type: 模型类型 (int8, fp16)
        similarity_threshold: 相似度阈值
        known_embeddings: 已知人脸特征向量（JSON字符串）
    
    Returns:
        Dict: 识别结果
    """
    try:
        # 解析已知特征向量
        try:
            known_embeddings_list = json.loads(known_embeddings)
            known_embeddings_array = [np.array(emb) for emb in known_embeddings_list]
        except:
            known_embeddings_array = []
        
        # 查找对应的量化模型
        models_dir = "quantization/mobile_models"
        detection_model_path = None
        recognition_model_path = None
        
        for filename in os.listdir(models_dir):
            if "face_detection" in filename and f"_{model_type}" in filename:
                detection_model_path = os.path.join(models_dir, filename)
            elif "face_recognition" in filename and f"_{model_type}" in filename:
                recognition_model_path = os.path.join(models_dir, filename)
        
        if detection_model_path is None or recognition_model_path is None:
            raise HTTPException(status_code=404, detail=f"未找到 {model_type} 类型的完整模型")
        
        # 初始化移动端服务
        mobile_service = MobileFaceService(
            detection_model_path=detection_model_path,
            recognition_model_path=recognition_model_path
        )
        
        # 保存上传的图像
        image_data = await image.read()
        
        # 执行人脸识别
        known_labels = [f"person_{i}" for i in range(len(known_embeddings_array))]
        results = mobile_service.recognize_faces(
            image_data, 
            known_embeddings_array, 
            known_labels,
            similarity_threshold
        )
        
        return JSONResponse(content={
            "message": "人脸识别完成",
            "model_type": model_type,
            "face_count": len(results),
            "results": results
        })
        
    except Exception as e:
        logging.error(f"移动端人脸识别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/benchmark_model")
async def benchmark_model(
    model_path: str = Form(...),
    test_images: List[UploadFile] = File(...),
    warmup_runs: int = Form(10),
    test_runs: int = Form(100)
):
    """
    模型性能基准测试
    
    Args:
        model_path: 模型文件路径
        test_images: 测试图像文件列表
        warmup_runs: 预热运行次数
        test_runs: 测试运行次数
    
    Returns:
        Dict: 性能测试结果
    """
    try:
        from mobile_inference import ONNXInference
        
        # 初始化推理引擎
        inference_engine = ONNXInference(model_path)
        
        # 准备测试数据
        test_inputs = []
        for image_file in test_images:
            image_data = await image_file.read()
            processed_image = inference_engine.preprocess_image(image_data)
            test_inputs.append(processed_image)
        
        # 执行性能测试
        performance_utils = PerformanceUtils()
        result = performance_utils.benchmark_model(
            model_path, test_inputs, warmup_runs, test_runs
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"性能测试失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model_info/{model_path:path}")
async def get_model_info(model_path: str):
    """
    获取模型信息
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        Dict: 模型信息
    """
    try:
        model_utils = ModelUtils()
        
        # 验证模型
        is_valid = model_utils.validate_onnx_model(model_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail="模型文件无效")
        
        # 获取模型信息
        info = model_utils.get_model_info(model_path)
        size_info = model_utils.get_model_size(model_path)
        
        result = {**info, **size_info}
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
