#!/usr/bin/env python3
"""
InsightFace OpenVINO 量化脚本
使用 OpenVINO 对 InsightFace 模型进行量化
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
from PIL import Image
import tqdm

# OpenVINO 导入
try:
    from openvino import Core, Model
    from openvino.tools import mo
    from openvino.runtime import compile_model
    from openvino.tools.pot import DataLoader, IEEngine, load_config, save_model
    from openvino.tools.pot.graph import load_model, save_model as pot_save_model
    from openvino.tools.pot.engines.ie_engine import IEEngine
    from openvino.tools.pot.graph.model_utils import compress_model_weights
    from openvino.tools.pot.pipeline.initializer import create_pipeline
    from openvino.tools.pot.statistics.collector import StatisticsCollector
    from openvino.tools.pot.utils.logger import init_logger
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    logger.warning("OpenVINO 未安装，请安装 openvino 和 openvino-dev")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenVINODataLoader(DataLoader):
    """OpenVINO 数据加载器"""
    
    def __init__(self, images_dir: str, input_name: str, input_shape: tuple):
        self.input_name = input_name
        self.input_shape = input_shape
        self.images = self._load_images(images_dir)
        self.current_index = 0
        
    def _load_images(self, images_dir: str) -> List[np.ndarray]:
        """加载图像"""
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
                        img = cv2.resize(img, (self.input_shape[2], self.input_shape[3]))
                        # 归一化到[0,1]
                        img = img.astype(np.float32) / 255.0
                        # 转换为NCHW格式
                        img = np.transpose(img, (2, 0, 1))
                        # 添加batch维度
                        img = np.expand_dims(img, axis=0)
                        images.append(img)
                except Exception as e:
                    logger.warning(f"无法加载图像 {img_path}: {e}")
        
        logger.info(f"加载了 {len(images)} 张图像")
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        if index >= len(self.images):
            raise IndexError("Index out of range")
        
        return {self.input_name: self.images[index]}

class OpenVINOQuantizer:
    """OpenVINO 模型量化器"""
    
    def __init__(self, output_dir: str = "models/openvino"):
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO 未安装，请安装 openvino 和 openvino-dev")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.core = Core()
        
    def convert_insightface_to_openvino(self, model_name: str, input_shape: tuple = (1, 3, 640, 640)) -> str:
        """将 InsightFace 模型转换为 OpenVINO 格式"""
        logger.info(f"开始转换 InsightFace 模型 {model_name} 到 OpenVINO...")
        
        try:
            # 注意：InsightFace 的模型结构可能需要特殊处理
            # 这里提供一个通用的转换示例
            logging.warning("注意：InsightFace 模型转换需要根据具体模型结构调整")
            
            # 这里需要根据实际的 InsightFace 模型来调整
            # 通常需要先将模型转换为 ONNX 格式，然后再转换为 OpenVINO
            
            # 示例：假设我们有一个 ONNX 模型
            onnx_path = f"models/onnx/{model_name}_fp32.onnx"
            if not os.path.exists(onnx_path):
                logger.error(f"ONNX 模型不存在: {onnx_path}")
                logger.info("请先运行 quantize_onnx.py 生成 ONNX 模型")
                return None
            
            # 转换为 OpenVINO IR
            ir_path = self.output_dir / f"{model_name}_fp32.xml"
            
            # 使用 Model Optimizer 转换
            mo.convert_model(
                input_model=onnx_path,
                output_dir=str(self.output_dir),
                model_name=f"{model_name}_fp32",
                input_shape=input_shape
            )
            
            logger.info(f"OpenVINO 模型已保存到: {ir_path}")
            return str(ir_path)
            
        except Exception as e:
            logger.error(f"模型转换失败: {e}")
            raise
    
    def quantize_int8(self, model_path: str, calibration_images_dir: str, 
                     input_name: str = "input", input_shape: tuple = (1, 3, 640, 640),
                     output_path: str = None) -> str:
        """INT8 量化"""
        logger.info(f"开始 INT8 量化: {model_path}")
        
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = self.output_dir / f"{model_name}_int8.xml"
        
        try:
            # 创建数据加载器
            data_loader = OpenVINODataLoader(calibration_images_dir, input_name, input_shape)
            
            # 加载模型
            model = self.core.read_model(model_path)
            
            # 配置量化参数
            engine_config = {
                "device": "CPU",
                "stat_requests_number": 4,
                "eval_requests_number": 4
            }
            
            # 创建引擎
            engine = IEEngine(config=engine_config, data_loader=data_loader)
            
            # 配置量化算法
            algorithms = [
                {
                    "name": "DefaultQuantization",
                    "params": {
                        "target_device": "CPU",
                        "preset": "performance",
                        "stat_subset_size": 300
                    }
                }
            ]
            
            # 创建量化管道
            pipeline = create_pipeline(algorithms, engine)
            
            # 执行量化
            compressed_model = pipeline.run(model)
            
            # 保存量化模型
            pot_save_model(compressed_model, str(self.output_dir), f"{Path(model_path).stem}_int8")
            
            logger.info(f"INT8 量化完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"INT8 量化失败: {e}")
            raise
    
    def quantize_fp16(self, model_path: str, output_path: str = None) -> str:
        """FP16 量化"""
        logger.info(f"开始 FP16 量化: {model_path}")
        
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = self.output_dir / f"{model_name}_fp16.xml"
        
        try:
            # 加载模型
            model = self.core.read_model(model_path)
            
            # 转换为 FP16
            model = mo.convert_model(
                input_model=model,
                compress_to_fp16=True
            )
            
            # 保存模型
            self.core.save_model(model, str(output_path))
            
            logger.info(f"FP16 量化完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"FP16 量化失败: {e}")
            raise
    
    def benchmark_model(self, model_path: str, input_shape: tuple = (1, 3, 640, 640), 
                       num_runs: int = 100) -> Dict[str, float]:
        """模型性能测试"""
        logger.info(f"开始性能测试: {model_path}")
        
        try:
            # 加载模型
            model = self.core.read_model(model_path)
            compiled_model = compile_model(model, "CPU")
            
            # 获取输入输出
            input_layer = compiled_model.input(0)
            output_layer = compiled_model.output(0)
            
            # 准备测试数据
            input_data = np.random.randn(*input_shape).astype(np.float32)
            
            # 预热
            for _ in range(10):
                compiled_model([input_data])
            
            # 性能测试
            import time
            times = []
            
            for _ in tqdm.tqdm(range(num_runs), desc="性能测试"):
                start_time = time.time()
                compiled_model([input_data])
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
        xml_path = Path(model_path)
        bin_path = xml_path.with_suffix('.bin')
        
        xml_size = os.path.getsize(xml_path) if xml_path.exists() else 0
        bin_size = os.path.getsize(bin_path) if bin_path.exists() else 0
        
        return xml_size + bin_size
    
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
    if not OPENVINO_AVAILABLE:
        logger.error("OpenVINO 未安装，请安装 openvino 和 openvino-dev")
        logger.info("安装命令: pip install openvino openvino-dev")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='InsightFace OpenVINO 量化工具')
    parser.add_argument('--model_name', type=str, default='buffalo_l', 
                       help='InsightFace 模型名称')
    parser.add_argument('--quantization_type', type=str, 
                       choices=['int8', 'fp16'], default='int8',
                       help='量化类型')
    parser.add_argument('--calibration_images', type=str, 
                       help='校准图像目录（INT8量化需要）')
    parser.add_argument('--input_shape', type=tuple, default=(1, 3, 640, 640),
                       help='输入形状')
    parser.add_argument('--output_dir', type=str, default='models/openvino',
                       help='输出目录')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能测试')
    
    args = parser.parse_args()
    
    # 创建量化器
    quantizer = OpenVINOQuantizer(args.output_dir)
    
    try:
        # 转换模型
        logger.info("步骤 1: 转换 InsightFace 模型到 OpenVINO")
        openvino_path = quantizer.convert_insightface_to_openvino(args.model_name, args.input_shape)
        
        if openvino_path is None:
            logger.error("模型转换失败")
            return
        
        # 量化模型
        logger.info(f"步骤 2: {args.quantization_type} 量化")
        if args.quantization_type == 'int8':
            if not args.calibration_images:
                logger.error("INT8 量化需要提供校准图像目录")
                return
            quantized_path = quantizer.quantize_int8(
                openvino_path, args.calibration_images, 
                input_shape=args.input_shape
            )
        elif args.quantization_type == 'fp16':
            quantized_path = quantizer.quantize_fp16(openvino_path)
        
        # 模型大小对比
        comparison = quantizer.compare_models(openvino_path, quantized_path)
        logger.info("模型大小对比:")
        logger.info(f"  原始模型: {comparison['original_size'] / 1024 / 1024:.2f} MB")
        logger.info(f"  量化模型: {comparison['quantized_size'] / 1024 / 1024:.2f} MB")
        logger.info(f"  压缩比: {comparison['compression_ratio']:.2f}x")
        logger.info(f"  大小减少: {comparison['size_reduction']:.1f}%")
        
        # 性能测试
        if args.benchmark:
            logger.info("步骤 3: 性能测试")
            original_stats = quantizer.benchmark_model(openvino_path, args.input_shape)
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
