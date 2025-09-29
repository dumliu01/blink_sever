#!/usr/bin/env python3
"""
InsightFace 量化工具使用示例
演示如何使用量化脚本和移动端推理代码
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantize_onnx import ONNXQuantizer
from quantize_tflite import TFLiteQuantizer
from quantize_openvino import OpenVINOQuantizer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_calibration_images(output_dir: str, num_images: int = 100):
    """创建校准图像"""
    import cv2
    import numpy as np
    
    calib_dir = Path(output_dir) / "calibration_images"
    calib_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"创建 {num_images} 张校准图像...")
    
    for i in range(num_images):
        # 生成随机图像
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 添加一些简单的几何形状
        cv2.circle(img, (320, 320), 100, (255, 255, 255), -1)
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)
        
        # 保存图像
        img_path = calib_dir / f"calib_{i:04d}.jpg"
        cv2.imwrite(str(img_path), img)
    
    logger.info(f"校准图像已保存到: {calib_dir}")
    return str(calib_dir)

def demo_onnx_quantization():
    """演示 ONNX 量化"""
    logger.info("=== ONNX 量化演示 ===")
    
    # 创建校准图像
    calib_dir = create_calibration_images("models/onnx", 50)
    
    # 创建量化器
    quantizer = ONNXQuantizer("models/onnx")
    
    try:
        # 转换模型
        logger.info("步骤 1: 转换 InsightFace 模型到 ONNX")
        onnx_path = quantizer.convert_insightface_to_onnx("buffalo_l")
        
        # 动态量化
        logger.info("步骤 2: 动态量化")
        dynamic_path = quantizer.quantize_dynamic(onnx_path)
        
        # 静态量化
        logger.info("步骤 3: 静态量化")
        static_path = quantizer.quantize_static(onnx_path, calib_dir)
        
        # 性能测试
        logger.info("步骤 4: 性能测试")
        original_stats = quantizer.benchmark_model(onnx_path)
        dynamic_stats = quantizer.benchmark_model(dynamic_path)
        static_stats = quantizer.benchmark_model(static_path)
        
        logger.info("ONNX 量化结果:")
        logger.info(f"  原始模型 FPS: {original_stats['fps']:.2f}")
        logger.info(f"  动态量化 FPS: {dynamic_stats['fps']:.2f}")
        logger.info(f"  静态量化 FPS: {static_stats['fps']:.2f}")
        
    except Exception as e:
        logger.error(f"ONNX 量化失败: {e}")

def demo_tflite_quantization():
    """演示 TensorFlow Lite 量化"""
    logger.info("=== TensorFlow Lite 量化演示 ===")
    
    # 创建校准图像
    calib_dir = create_calibration_images("models/tflite", 50)
    
    # 创建量化器
    quantizer = TFLiteQuantizer("models/tflite")
    
    try:
        # 转换模型
        logger.info("步骤 1: 转换 InsightFace 模型到 TensorFlow Lite")
        tflite_path = quantizer.convert_insightface_to_tflite("buffalo_l")
        
        # INT8 量化
        logger.info("步骤 2: INT8 量化")
        calibration_images = quantizer.load_calibration_images(calib_dir, (1, 640, 640, 3))
        int8_path = quantizer.quantize_post_training_int8(tflite_path, calibration_images)
        
        # Float16 量化
        logger.info("步骤 3: Float16 量化")
        float16_path = quantizer.quantize_post_training_float16(tflite_path)
        
        # 模型大小对比
        int8_comparison = quantizer.compare_models(tflite_path, int8_path)
        float16_comparison = quantizer.compare_models(tflite_path, float16_path)
        
        logger.info("TensorFlow Lite 量化结果:")
        logger.info(f"  INT8 压缩比: {int8_comparison['compression_ratio']:.2f}x")
        logger.info(f"  Float16 压缩比: {float16_comparison['compression_ratio']:.2f}x")
        
    except Exception as e:
        logger.error(f"TensorFlow Lite 量化失败: {e}")

def demo_openvino_quantization():
    """演示 OpenVINO 量化"""
    logger.info("=== OpenVINO 量化演示 ===")
    
    try:
        # 创建校准图像
        calib_dir = create_calibration_images("models/openvino", 50)
        
        # 创建量化器
        quantizer = OpenVINOQuantizer("models/openvino")
        
        # 转换模型
        logger.info("步骤 1: 转换 InsightFace 模型到 OpenVINO")
        openvino_path = quantizer.convert_insightface_to_openvino("buffalo_l")
        
        if openvino_path:
            # INT8 量化
            logger.info("步骤 2: INT8 量化")
            int8_path = quantizer.quantize_int8(openvino_path, calib_dir)
            
            # FP16 量化
            logger.info("步骤 3: FP16 量化")
            fp16_path = quantizer.quantize_fp16(openvino_path)
            
            # 模型大小对比
            int8_comparison = quantizer.compare_models(openvino_path, int8_path)
            fp16_comparison = quantizer.compare_models(openvino_path, fp16_path)
            
            logger.info("OpenVINO 量化结果:")
            logger.info(f"  INT8 压缩比: {int8_comparison['compression_ratio']:.2f}x")
            logger.info(f"  FP16 压缩比: {fp16_comparison['compression_ratio']:.2f}x")
        
    except Exception as e:
        logger.error(f"OpenVINO 量化失败: {e}")

def demo_mobile_inference():
    """演示移动端推理"""
    logger.info("=== 移动端推理演示 ===")
    
    # 这里演示如何使用移动端推理代码
    logger.info("移动端推理代码已生成:")
    logger.info("  iOS: mobile_inference/ios/InsightFaceInference.swift")
    logger.info("  Android: mobile_inference/android/InsightFaceInference.kt")
    logger.info("  Android Java: mobile_inference/android/InsightFaceInference.java")
    
    logger.info("\n使用说明:")
    logger.info("1. 将量化后的模型文件添加到移动项目中")
    logger.info("2. 集成相应的推理代码")
    logger.info("3. 配置模型路径和参数")
    logger.info("4. 调用 detectFaces() 或 recognizeFace() 方法")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='InsightFace 量化工具演示')
    parser.add_argument('--demo', type=str, 
                       choices=['onnx', 'tflite', 'openvino', 'mobile', 'all'],
                       default='all', help='演示类型')
    
    args = parser.parse_args()
    
    logger.info("开始 InsightFace 量化工具演示...")
    
    if args.demo in ['onnx', 'all']:
        demo_onnx_quantization()
    
    if args.demo in ['tflite', 'all']:
        demo_tflite_quantization()
    
    if args.demo in ['openvino', 'all']:
        demo_openvino_quantization()
    
    if args.demo in ['mobile', 'all']:
        demo_mobile_inference()
    
    logger.info("演示完成！")

if __name__ == "__main__":
    main()
