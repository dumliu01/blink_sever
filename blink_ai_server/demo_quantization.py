"""
量化功能演示脚本
展示如何使用量化功能进行模型转换、量化和移动端推理
"""

import os
import sys
import numpy as np
import logging
from PIL import Image
import cv2

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantization import ModelConverter, ModelQuantizer
from quantization.utils import ModelUtils, PerformanceUtils
from mobile_inference import ONNXInference, MobileFaceService

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_demo_images(output_dir: str, num_images: int = 10):
    """创建演示图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"创建 {num_images} 张演示图像...")
    
    for i in range(num_images):
        # 创建随机图像
        img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = os.path.join(output_dir, f"demo_{i:03d}.jpg")
        img.save(img_path)
    
    logger.info(f"演示图像已保存到: {output_dir}")

def demo_model_conversion():
    """演示模型转换"""
    logger.info("=== 模型转换演示 ===")
    
    try:
        # 初始化模型转换器
        converter = ModelConverter("quantization/mobile_models")
        
        # 转换InsightFace模型（需要先安装InsightFace）
        logger.info("开始转换InsightFace模型...")
        result = converter.convert_insightface_to_onnx(
            model_name="buffalo_l",
            input_size=(640, 640)
        )
        
        if result.get("success", False):
            logger.info("模型转换成功!")
            logger.info(f"检测模型: {result['detection_model']['path']}")
            logger.info(f"识别模型: {result['recognition_model']['path']}")
        else:
            logger.error(f"模型转换失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        logger.error(f"模型转换演示失败: {e}")

def demo_model_quantization():
    """演示模型量化"""
    logger.info("=== 模型量化演示 ===")
    
    try:
        # 初始化模型量化器
        quantizer = ModelQuantizer("quantization/mobile_models")
        
        # 创建校准数据集
        calibration_dir = "quantization/datasets/calibration_images"
        create_demo_images(calibration_dir, 20)
        
        # 查找ONNX模型文件
        models_dir = "quantization/mobile_models"
        onnx_models = []
        
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith('.onnx') and not any(x in filename for x in ['_int8', '_fp16', '_dynamic']):
                    onnx_models.append(os.path.join(models_dir, filename))
        
        if not onnx_models:
            logger.warning("未找到ONNX模型文件，跳过量化演示")
            return
        
        # 对每个模型进行量化
        for model_path in onnx_models:
            logger.info(f"量化模型: {model_path}")
            
            # INT8量化
            logger.info("执行INT8量化...")
            int8_result = quantizer.quantize_to_int8(model_path, calibration_dir)
            if int8_result.get("success", False):
                logger.info(f"INT8量化成功: {int8_result['compression_ratio']:.2f}x 压缩")
            
            # FP16量化
            logger.info("执行FP16量化...")
            fp16_result = quantizer.quantize_to_fp16(model_path)
            if fp16_result.get("success", False):
                logger.info(f"FP16量化成功: {fp16_result['compression_ratio']:.2f}x 压缩")
            
            # 动态INT8量化
            logger.info("执行动态INT8量化...")
            dynamic_result = quantizer.quantize_dynamic_int8(model_path)
            if dynamic_result.get("success", False):
                logger.info(f"动态INT8量化成功: {dynamic_result['compression_ratio']:.2f}x 压缩")
            
    except Exception as e:
        logger.error(f"模型量化演示失败: {e}")

def demo_mobile_inference():
    """演示移动端推理"""
    logger.info("=== 移动端推理演示 ===")
    
    try:
        # 查找量化模型
        models_dir = "quantization/mobile_models"
        detection_model = None
        recognition_model = None
        
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if "face_detection" in filename and "_int8" in filename:
                    detection_model = os.path.join(models_dir, filename)
                elif "face_recognition" in filename and "_int8" in filename:
                    recognition_model = os.path.join(models_dir, filename)
        
        if not detection_model or not recognition_model:
            logger.warning("未找到量化模型文件，跳过移动端推理演示")
            return
        
        # 初始化移动端服务
        logger.info("初始化移动端人脸识别服务...")
        mobile_service = MobileFaceService(
            detection_model_path=detection_model,
            recognition_model_path=recognition_model
        )
        
        # 创建测试图像
        test_image_dir = "test_images"
        create_demo_images(test_image_dir, 3)
        
        # 测试人脸检测
        logger.info("测试人脸检测...")
        test_image_path = os.path.join(test_image_dir, "demo_000.jpg")
        
        faces = mobile_service.detect_faces(test_image_path)
        logger.info(f"检测到 {len(faces)} 个人脸")
        
        # 测试人脸识别
        logger.info("测试人脸识别...")
        known_embeddings = []
        known_labels = []
        
        # 使用前几张图像作为已知人脸
        for i in range(min(2, len(os.listdir(test_image_dir)))):
            img_path = os.path.join(test_image_dir, f"demo_{i:03d}.jpg")
            embedding = mobile_service.extract_face_embedding(img_path)
            if embedding is not None:
                known_embeddings.append(embedding)
                known_labels.append(f"person_{i}")
        
        if known_embeddings:
            results = mobile_service.recognize_faces(
                test_image_path,
                known_embeddings,
                known_labels,
                similarity_threshold=0.6
            )
            logger.info(f"识别结果: {len(results)} 个有效人脸")
        
    except Exception as e:
        logger.error(f"移动端推理演示失败: {e}")

def demo_performance_comparison():
    """演示性能对比"""
    logger.info("=== 性能对比演示 ===")
    
    try:
        # 查找不同精度的模型
        models_dir = "quantization/mobile_models"
        models = []
        
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith('.onnx'):
                    model_path = os.path.join(models_dir, filename)
                    if "_int8" in filename:
                        models.append({"name": "INT8", "path": model_path})
                    elif "_fp16" in filename:
                        models.append({"name": "FP16", "path": model_path})
                    elif "_dynamic_int8" in filename:
                        models.append({"name": "Dynamic INT8", "path": model_path})
                    elif not any(x in filename for x in ['_int8', '_fp16', '_dynamic']):
                        models.append({"name": "Original", "path": model_path})
        
        if len(models) < 2:
            logger.warning("模型数量不足，跳过性能对比演示")
            return
        
        # 创建测试数据
        test_inputs = [np.random.randn(1, 3, 640, 640).astype(np.float32) for _ in range(5)]
        
        # 性能对比
        performance_utils = PerformanceUtils()
        comparison_result = performance_utils.compare_performance(models, test_inputs)
        
        if comparison_result.get("success", False):
            logger.info("性能对比结果:")
            for model in comparison_result["models"]:
                logger.info(f"{model['name']}: {model['fps']:.2f} FPS, {model['avg_time']:.4f}s")
        
    except Exception as e:
        logger.error(f"性能对比演示失败: {e}")

def main():
    """主函数"""
    logger.info("开始量化功能演示...")
    
    # 创建必要的目录
    os.makedirs("quantization/mobile_models", exist_ok=True)
    os.makedirs("quantization/datasets/calibration_images", exist_ok=True)
    os.makedirs("test_images", exist_ok=True)
    
    # 运行演示
    demo_model_conversion()
    demo_model_quantization()
    demo_mobile_inference()
    demo_performance_comparison()
    
    logger.info("量化功能演示完成!")

if __name__ == "__main__":
    main()
