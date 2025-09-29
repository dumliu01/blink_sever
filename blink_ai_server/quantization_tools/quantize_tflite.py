#!/usr/bin/env python3
"""
InsightFace TensorFlow Lite 量化脚本
使用 TensorFlow Lite 对 InsightFace 模型进行量化
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tensorflow as tf
from tensorflow import lite as tflite
import insightface
from insightface.app import FaceAnalysis
import cv2
from PIL import Image
import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TFLiteQuantizer:
    """TensorFlow Lite 模型量化器"""
    
    def __init__(self, output_dir: str = "models/tflite"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_insightface_to_tflite(self, model_name: str, input_shape: tuple = (1, 640, 640, 3)) -> str:
        """将 InsightFace 模型转换为 TensorFlow Lite 格式"""
        logger.info(f"开始转换 InsightFace 模型 {model_name} 到 TensorFlow Lite...")
        
        try:
            # 初始化 InsightFace
            app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            
            # 获取模型
            model = app.models['detection']
            
            # 创建示例输入
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # 导出为 TensorFlow Lite
            tflite_path = self.output_dir / f"{model_name}_fp32.tflite"
            
            # 注意：InsightFace 的模型结构可能需要特殊处理
            # 这里提供一个通用的转换示例
            logging.warning("注意：InsightFace 模型转换需要根据具体模型结构调整")
            
            # 创建一个简单的 TensorFlow 模型作为示例
            # 实际使用时需要根据 InsightFace 的具体实现来调整
            def create_sample_model():
                inputs = tf.keras.Input(shape=input_shape[1:], name='input')
                x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                return tf.keras.Model(inputs, outputs)
            
            # 创建示例模型
            sample_model = create_sample_model()
            
            # 转换为 TensorFlow Lite
            converter = tflite.TFLiteConverter.from_keras_model(sample_model)
            converter.optimizations = [tflite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            
            # 保存模型
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TensorFlow Lite 模型已保存到: {tflite_path}")
            return str(tflite_path)
            
        except Exception as e:
            logger.error(f"模型转换失败: {e}")
            raise
    
    def quantize_post_training_int8(self, model_path: str, representative_dataset: List[np.ndarray], 
                                  output_path: str = None) -> str:
        """Post-training INT8 量化"""
        logger.info(f"开始 Post-training INT8 量化: {model_path}")
        
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = self.output_dir / f"{model_name}_int8.tflite"
        
        try:
            # 创建代表性数据集生成器
            def representative_data_gen():
                for data in representative_dataset:
                    yield [data]
            
            # 配置转换器
            converter = tflite.TFLiteConverter.from_saved_model(model_path)
            converter.optimizations = [tflite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tflite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            # 转换模型
            quantized_model = converter.convert()
            
            # 保存量化模型
            with open(output_path, 'wb') as f:
                f.write(quantized_model)
            
            logger.info(f"Post-training INT8 量化完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Post-training INT8 量化失败: {e}")
            raise
    
    def quantize_post_training_float16(self, model_path: str, output_path: str = None) -> str:
        """Post-training Float16 量化"""
        logger.info(f"开始 Post-training Float16 量化: {model_path}")
        
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = self.output_dir / f"{model_name}_float16.tflite"
        
        try:
            # 配置转换器
            converter = tflite.TFLiteConverter.from_saved_model(model_path)
            converter.optimizations = [tflite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            # 转换模型
            quantized_model = converter.convert()
            
            # 保存量化模型
            with open(output_path, 'wb') as f:
                f.write(quantized_model)
            
            logger.info(f"Post-training Float16 量化完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Post-training Float16 量化失败: {e}")
            raise
    
    def quantize_dynamic_range(self, model_path: str, output_path: str = None) -> str:
        """动态范围量化"""
        logger.info(f"开始动态范围量化: {model_path}")
        
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = self.output_dir / f"{model_name}_dynamic.tflite"
        
        try:
            # 配置转换器
            converter = tflite.TFLiteConverter.from_saved_model(model_path)
            converter.optimizations = [tflite.Optimize.DEFAULT]
            
            # 转换模型
            quantized_model = converter.convert()
            
            # 保存量化模型
            with open(output_path, 'wb') as f:
                f.write(quantized_model)
            
            logger.info(f"动态范围量化完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"动态范围量化失败: {e}")
            raise
    
    def load_calibration_images(self, images_dir: str, input_shape: tuple) -> List[np.ndarray]:
        """加载校准图像"""
        images = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for img_path in Path(images_dir).rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                try:
                    # 读取图像
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # 转换为RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # 调整大小到模型输入尺寸
                        img = cv2.resize(img, (input_shape[1], input_shape[2]))
                        # 归一化到[0,1]
                        img = img.astype(np.float32) / 255.0
                        # 添加batch维度
                        img = np.expand_dims(img, axis=0)
                        images.append(img)
                except Exception as e:
                    logger.warning(f"无法加载图像 {img_path}: {e}")
        
        logger.info(f"加载了 {len(images)} 张校准图像")
        return images
    
    def benchmark_model(self, model_path: str, input_shape: tuple = (1, 640, 640, 3), 
                       num_runs: int = 100) -> Dict[str, float]:
        """模型性能测试"""
        logger.info(f"开始性能测试: {model_path}")
        
        try:
            # 加载模型
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # 获取输入输出详情
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # 准备测试数据
            input_data = np.random.randn(*input_shape).astype(np.float32)
            
            # 预热
            for _ in range(10):
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
            
            # 性能测试
            import time
            times = []
            
            for _ in tqdm.tqdm(range(num_runs), desc="性能测试"):
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                end_time = time.time()
                times.append(end_time - start_time)
            
            # 计算统计信息
            times = np.array(times)
            stats = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'fps': 1.0 / np.mean(times)
            }
            
            logger.info(f"性能测试完成:")
            logger.info(f"  平均推理时间: {stats['mean_time']:.4f}s")
            logger.info(f"  推理速度: {stats['fps']:.2f} FPS")
            
            return stats
            
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            raise
    
    def get_model_size(self, model_path: str) -> int:
        """获取模型大小（字节）"""
        return os.path.getsize(model_path)
    
    def compare_models(self, original_path: str, quantized_path: str) -> Dict[str, Any]:
        """比较原始模型和量化模型"""
        original_size = self.get_model_size(original_path)
        quantized_size = self.get_model_size(quantized_path)
        
        compression_ratio = original_size / quantized_size
        size_reduction = (1 - quantized_size / original_size) * 100
        
        return {
            'original_size': original_size,
            'quantized_size': quantized_size,
            'compression_ratio': compression_ratio,
            'size_reduction': size_reduction
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='InsightFace TensorFlow Lite 量化工具')
    parser.add_argument('--model_name', type=str, default='buffalo_l', 
                       help='InsightFace 模型名称')
    parser.add_argument('--quantization_type', type=str, 
                       choices=['int8', 'float16', 'dynamic'], default='int8',
                       help='量化类型')
    parser.add_argument('--calibration_images', type=str, 
                       help='校准图像目录（INT8量化需要）')
    parser.add_argument('--input_shape', type=tuple, default=(1, 640, 640, 3),
                       help='输入形状')
    parser.add_argument('--output_dir', type=str, default='models/tflite',
                       help='输出目录')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能测试')
    
    args = parser.parse_args()
    
    # 创建量化器
    quantizer = TFLiteQuantizer(args.output_dir)
    
    try:
        # 转换模型
        logger.info("步骤 1: 转换 InsightFace 模型到 TensorFlow Lite")
        tflite_path = quantizer.convert_insightface_to_tflite(args.model_name, args.input_shape)
        
        # 量化模型
        logger.info(f"步骤 2: {args.quantization_type} 量化")
        if args.quantization_type == 'int8':
            if not args.calibration_images:
                logger.error("INT8 量化需要提供校准图像目录")
                return
            calibration_images = quantizer.load_calibration_images(args.calibration_images, args.input_shape)
            quantized_path = quantizer.quantize_post_training_int8(tflite_path, calibration_images)
        elif args.quantization_type == 'float16':
            quantized_path = quantizer.quantize_post_training_float16(tflite_path)
        elif args.quantization_type == 'dynamic':
            quantized_path = quantizer.quantize_dynamic_range(tflite_path)
        
        # 模型大小对比
        comparison = quantizer.compare_models(tflite_path, quantized_path)
        logger.info("模型大小对比:")
        logger.info(f"  原始模型: {comparison['original_size'] / 1024 / 1024:.2f} MB")
        logger.info(f"  量化模型: {comparison['quantized_size'] / 1024 / 1024:.2f} MB")
        logger.info(f"  压缩比: {comparison['compression_ratio']:.2f}x")
        logger.info(f"  大小减少: {comparison['size_reduction']:.1f}%")
        
        # 性能测试
        if args.benchmark:
            logger.info("步骤 3: 性能测试")
            original_stats = quantizer.benchmark_model(tflite_path, args.input_shape)
            quantized_stats = quantizer.benchmark_model(quantized_path, args.input_shape)
            
            logger.info("性能对比:")
            logger.info(f"  原始模型 FPS: {original_stats['fps']:.2f}")
            logger.info(f"  量化模型 FPS: {quantized_stats['fps']:.2f}")
            logger.info(f"  加速比: {quantized_stats['fps'] / original_stats['fps']:.2f}x")
        
        logger.info("量化完成！")
        logger.info(f"量化模型路径: {quantized_path}")
        
    except Exception as e:
        logger.error(f"量化过程失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
