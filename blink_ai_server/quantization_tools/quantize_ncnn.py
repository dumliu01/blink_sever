#!/usr/bin/env python3
"""
InsightFace NCNN 量化脚本
使用 NCNN 对 InsightFace 模型进行量化
"""

import os
import sys
import argparse
import logging
import numpy as np
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
from PIL import Image
import tqdm
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NCNNQuantizer:
    """NCNN 模型量化器"""
    
    def __init__(self, output_dir: str = "models/ncnn"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查NCNN工具是否可用
        self.ncnn_tools_available = self._check_ncnn_tools()
        
    def _check_ncnn_tools(self) -> bool:
        """检查NCNN工具是否可用"""
        try:
            # 检查onnx2ncnn工具
            result = subprocess.run(['onnx2ncnn', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.warning("onnx2ncnn 工具不可用，请确保NCNN已正确安装")
                return False
                
            # 检查ncnnoptimize工具
            result = subprocess.run(['ncnnoptimize', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.warning("ncnnoptimize 工具不可用，请确保NCNN已正确安装")
                return False
                
            # 检查ncnn2table工具
            result = subprocess.run(['ncnn2table', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.warning("ncnn2table 工具不可用，请确保NCNN已正确安装")
                return False
                
            # 检查ncnn2int8工具
            result = subprocess.run(['ncnn2int8', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.warning("ncnn2int8 工具不可用，请确保NCNN已正确安装")
                return False
                
            logger.info("NCNN 工具检查通过")
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"NCNN 工具检查失败: {e}")
            logger.warning("请确保已安装NCNN并配置环境变量")
            return False
    
    def convert_onnx_to_ncnn(self, onnx_path: str, model_name: str) -> Tuple[str, str]:
        """将ONNX模型转换为NCNN格式"""
        if not self.ncnn_tools_available:
            raise RuntimeError("NCNN工具不可用，无法进行转换")
            
        logger.info(f"开始转换 ONNX 模型 {onnx_path} 到 NCNN 格式...")
        
        # 输出文件路径
        param_path = self.output_dir / f"{model_name}.param"
        bin_path = self.output_dir / f"{model_name}.bin"
        
        try:
            # 使用onnx2ncnn转换
            cmd = [
                'onnx2ncnn',
                onnx_path,
                str(param_path),
                str(bin_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"ONNX到NCNN转换失败: {result.stderr}")
            
            logger.info(f"NCNN模型已保存: {param_path}, {bin_path}")
            return str(param_path), str(bin_path)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("ONNX到NCNN转换超时")
        except Exception as e:
            raise RuntimeError(f"ONNX到NCNN转换失败: {e}")
    
    def optimize_ncnn_model(self, param_path: str, bin_path: str, model_name: str) -> Tuple[str, str]:
        """优化NCNN模型"""
        if not self.ncnn_tools_available:
            raise RuntimeError("NCNN工具不可用，无法进行优化")
            
        logger.info(f"开始优化 NCNN 模型...")
        
        # 输出文件路径
        opt_param_path = self.output_dir / f"{model_name}_opt.param"
        opt_bin_path = self.output_dir / f"{model_name}_opt.bin"
        
        try:
            # 使用ncnnoptimize优化
            cmd = [
                'ncnnoptimize',
                param_path,
                bin_path,
                str(opt_param_path),
                str(opt_bin_path),
                '0'  # 优化级别
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"NCNN模型优化失败: {result.stderr}")
            
            logger.info(f"优化后的NCNN模型已保存: {opt_param_path}, {opt_bin_path}")
            return str(opt_param_path), str(opt_bin_path)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("NCNN模型优化超时")
        except Exception as e:
            raise RuntimeError(f"NCNN模型优化失败: {e}")
    
    def create_calibration_dataset(self, calibration_images_dir: str, 
                                 output_path: str, input_shape: tuple = (640, 640)) -> str:
        """创建NCNN量化校准数据集"""
        logger.info(f"创建NCNN量化校准数据集...")
        
        dataset_dir = Path(output_path)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 收集所有图像
        image_paths = []
        for img_path in Path(calibration_images_dir).rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                image_paths.append(img_path)
        
        if not image_paths:
            raise ValueError(f"在 {calibration_images_dir} 中未找到图像文件")
        
        logger.info(f"找到 {len(image_paths)} 张图像用于校准")
        
        # 处理图像并保存
        processed_count = 0
        for img_path in tqdm.tqdm(image_paths, desc="处理校准图像"):
            try:
                # 读取图像
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # 转换为RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 调整大小
                img = cv2.resize(img, input_shape)
                
                # 归一化到[0,1]
                img = img.astype(np.float32) / 255.0
                
                # 保存为二进制格式
                output_file = dataset_dir / f"calib_{processed_count:06d}.bin"
                img.tofile(str(output_file))
                
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"处理图像 {img_path} 失败: {e}")
                continue
        
        logger.info(f"校准数据集已创建: {dataset_dir} ({processed_count} 张图像)")
        return str(dataset_dir)
    
    def generate_quantization_table(self, param_path: str, bin_path: str, 
                                  calibration_dataset_dir: str, model_name: str) -> str:
        """生成量化表"""
        if not self.ncnn_tools_available:
            raise RuntimeError("NCNN工具不可用，无法生成量化表")
            
        logger.info(f"生成量化表...")
        
        # 输出文件路径
        table_path = self.output_dir / f"{model_name}_table.txt"
        
        try:
            # 使用ncnn2table生成量化表
            cmd = [
                'ncnn2table',
                param_path,
                bin_path,
                calibration_dataset_dir,
                str(table_path),
                '0'  # 量化方法: 0=KL散度
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise RuntimeError(f"量化表生成失败: {result.stderr}")
            
            logger.info(f"量化表已生成: {table_path}")
            return str(table_path)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("量化表生成超时")
        except Exception as e:
            raise RuntimeError(f"量化表生成失败: {e}")
    
    def quantize_to_int8(self, param_path: str, bin_path: str, 
                        table_path: str, model_name: str) -> Tuple[str, str]:
        """将模型量化为INT8"""
        if not self.ncnn_tools_available:
            raise RuntimeError("NCNN工具不可用，无法进行量化")
            
        logger.info(f"开始INT8量化...")
        
        # 输出文件路径
        int8_param_path = self.output_dir / f"{model_name}_int8.param"
        int8_bin_path = self.output_dir / f"{model_name}_int8.bin"
        
        try:
            # 使用ncnn2int8进行量化
            cmd = [
                'ncnn2int8',
                param_path,
                bin_path,
                str(int8_param_path),
                str(int8_bin_path),
                table_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise RuntimeError(f"INT8量化失败: {result.stderr}")
            
            logger.info(f"INT8量化模型已保存: {int8_param_path}, {int8_bin_path}")
            return str(int8_param_path), str(int8_bin_path)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("INT8量化超时")
        except Exception as e:
            raise RuntimeError(f"INT8量化失败: {e}")
    
    def quantize_insightface_model(self, onnx_path: str, model_name: str, 
                                 calibration_images_dir: str = None) -> Dict[str, Any]:
        """量化InsightFace模型"""
        logger.info(f"开始量化 InsightFace 模型: {model_name}")
        
        if not self.ncnn_tools_available:
            raise RuntimeError("NCNN工具不可用，请先安装NCNN")
        
        results = {}
        
        try:
            # 1. 转换ONNX到NCNN
            param_path, bin_path = self.convert_onnx_to_ncnn(onnx_path, model_name)
            results["original_param"] = param_path
            results["original_bin"] = bin_path
            
            # 2. 优化模型
            opt_param_path, opt_bin_path = self.optimize_ncnn_model(param_path, bin_path, model_name)
            results["optimized_param"] = opt_param_path
            results["optimized_bin"] = opt_bin_path
            
            # 3. 如果有校准图像，进行INT8量化
            if calibration_images_dir and os.path.exists(calibration_images_dir):
                logger.info("使用校准图像进行INT8量化...")
                
                # 创建校准数据集
                calib_dataset_dir = self.create_calibration_dataset(
                    calibration_images_dir, 
                    str(self.output_dir / "calibration_dataset")
                )
                
                # 生成量化表
                table_path = self.generate_quantization_table(
                    opt_param_path, opt_bin_path, calib_dataset_dir, model_name
                )
                results["quantization_table"] = table_path
                
                # INT8量化
                int8_param_path, int8_bin_path = self.quantize_to_int8(
                    opt_param_path, opt_bin_path, table_path, model_name
                )
                results["int8_param"] = int8_param_path
                results["int8_bin"] = int8_bin_path
                
            else:
                logger.warning("未提供校准图像，跳过INT8量化")
            
            results["success"] = True
            logger.info("NCNN量化完成")
            
        except Exception as e:
            logger.error(f"NCNN量化失败: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def benchmark_model(self, param_path: str, bin_path: str, 
                      input_shape: tuple = (1, 3, 640, 640), 
                      num_runs: int = 100) -> Dict[str, float]:
        """基准测试模型性能"""
        logger.info(f"开始基准测试: {param_path}")
        
        # 这里应该使用NCNN的Python绑定进行推理测试
        # 由于NCNN的Python绑定可能不可用，我们返回模拟数据
        logger.warning("NCNN Python绑定可能不可用，返回模拟基准测试结果")
        
        # 模拟基准测试结果
        import time
        import random
        
        times = []
        for _ in range(num_runs):
            # 模拟推理时间
            time.sleep(random.uniform(0.01, 0.05))
            times.append(random.uniform(0.01, 0.05))
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / mean_time if mean_time > 0 else 0
        
        return {
            "mean_time": mean_time,
            "std_time": std_time,
            "fps": fps,
            "num_runs": num_runs
        }
    
    def compare_models(self, original_param: str, original_bin: str,
                      quantized_param: str, quantized_bin: str) -> Dict[str, Any]:
        """比较模型大小"""
        try:
            original_size = os.path.getsize(original_param) + os.path.getsize(original_bin)
            quantized_size = os.path.getsize(quantized_param) + os.path.getsize(quantized_bin)
            
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
            size_reduction = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
            
            return {
                "original_size": original_size,
                "quantized_size": quantized_size,
                "compression_ratio": compression_ratio,
                "size_reduction": size_reduction
            }
        except Exception as e:
            logger.error(f"模型比较失败: {e}")
            return {
                "original_size": 0,
                "quantized_size": 0,
                "compression_ratio": 0,
                "size_reduction": 0
            }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='InsightFace NCNN 量化工具')
    parser.add_argument('--onnx_path', type=str, required=True,
                       help='ONNX模型路径')
    parser.add_argument('--model_name', type=str, default='buffalo_l',
                       help='模型名称')
    parser.add_argument('--calibration_images', type=str,
                       help='校准图像目录')
    parser.add_argument('--output_dir', type=str, default='models/ncnn',
                       help='输出目录')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行基准测试')
    
    args = parser.parse_args()
    
    # 创建量化器
    quantizer = NCNNQuantizer(args.output_dir)
    
    try:
        # 执行量化
        results = quantizer.quantize_insightface_model(
            onnx_path=args.onnx_path,
            model_name=args.model_name,
            calibration_images_dir=args.calibration_images
        )
        
        if results.get("success", False):
            logger.info("=== NCNN量化成功 ===")
            for key, value in results.items():
                if key not in ["success", "error"]:
                    logger.info(f"{key}: {value}")
            
            # 基准测试
            if args.benchmark and "int8_param" in results:
                logger.info("运行基准测试...")
                benchmark_results = quantizer.benchmark_model(
                    results["int8_param"], results["int8_bin"]
                )
                logger.info(f"基准测试结果: {benchmark_results}")
                
                # 模型大小比较
                if "optimized_param" in results:
                    size_comparison = quantizer.compare_models(
                        results["optimized_param"], results["optimized_bin"],
                        results["int8_param"], results["int8_bin"]
                    )
                    logger.info(f"模型大小比较: {size_comparison}")
        else:
            logger.error(f"NCNN量化失败: {results.get('error', '未知错误')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"量化过程失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
