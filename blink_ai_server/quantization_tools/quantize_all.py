#!/usr/bin/env python3
"""
InsightFace 统一量化脚本
支持 ONNX、TensorFlow Lite、OpenVINO 三种量化方案
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantize_onnx import ONNXQuantizer
from quantize_tflite import TFLiteQuantizer
from quantize_openvino import OpenVINOQuantizer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedQuantizer:
    """统一量化器"""
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建各格式的输出目录
        (self.output_dir / "onnx").mkdir(exist_ok=True)
        (self.output_dir / "tflite").mkdir(exist_ok=True)
        (self.output_dir / "openvino").mkdir(exist_ok=True)
        
        # 初始化各量化器
        self.onnx_quantizer = ONNXQuantizer(str(self.output_dir / "onnx"))
        self.tflite_quantizer = TFLiteQuantizer(str(self.output_dir / "tflite"))
        
        try:
            self.openvino_quantizer = OpenVINOQuantizer(str(self.output_dir / "openvino"))
        except ImportError:
            logger.warning("OpenVINO 未安装，跳过 OpenVINO 量化")
            self.openvino_quantizer = None
    
    def create_calibration_images(self, num_images: int = 100) -> str:
        """创建校准图像"""
        calib_dir = self.output_dir / "calibration_images"
        calib_dir.mkdir(exist_ok=True)
        
        logger.info(f"创建 {num_images} 张校准图像...")
        
        import cv2
        import numpy as np
        
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
    
    def quantize_all(self, model_name: str = "buffalo_l", 
                    quantization_types: List[str] = None,
                    create_calibration: bool = True,
                    benchmark: bool = True) -> Dict[str, Any]:
        """执行所有量化"""
        
        if quantization_types is None:
            quantization_types = ["onnx", "tflite"]
            if self.openvino_quantizer is not None:
                quantization_types.append("openvino")
        
        results = {}
        calib_dir = None
        
        # 创建校准图像
        if create_calibration:
            calib_dir = self.create_calibration_images()
        
        # ONNX 量化
        if "onnx" in quantization_types:
            logger.info("=== 开始 ONNX 量化 ===")
            try:
                onnx_results = self.quantize_onnx(model_name, calib_dir, benchmark)
                results["onnx"] = onnx_results
            except Exception as e:
                logger.error(f"ONNX 量化失败: {e}")
                results["onnx"] = {"error": str(e)}
        
        # TensorFlow Lite 量化
        if "tflite" in quantization_types:
            logger.info("=== 开始 TensorFlow Lite 量化 ===")
            try:
                tflite_results = self.quantize_tflite(model_name, calib_dir, benchmark)
                results["tflite"] = tflite_results
            except Exception as e:
                logger.error(f"TensorFlow Lite 量化失败: {e}")
                results["tflite"] = {"error": str(e)}
        
        # OpenVINO 量化
        if "openvino" in quantization_types and self.openvino_quantizer is not None:
            logger.info("=== 开始 OpenVINO 量化 ===")
            try:
                openvino_results = self.quantize_openvino(model_name, calib_dir, benchmark)
                results["openvino"] = openvino_results
            except Exception as e:
                logger.error(f"OpenVINO 量化失败: {e}")
                results["openvino"] = {"error": str(e)}
        
        return results
    
    def quantize_onnx(self, model_name: str, calib_dir: str, benchmark: bool) -> Dict[str, Any]:
        """ONNX 量化"""
        start_time = time.time()
        
        # 转换模型
        onnx_path = self.onnx_quantizer.convert_insightface_to_onnx(model_name)
        
        # 动态量化
        dynamic_path = self.onnx_quantizer.quantize_dynamic(onnx_path)
        
        # 静态量化
        static_path = None
        if calib_dir:
            static_path = self.onnx_quantizer.quantize_static(onnx_path, calib_dir)
        
        # 性能测试
        benchmark_results = {}
        if benchmark:
            benchmark_results["original"] = self.onnx_quantizer.benchmark_model(onnx_path)
            benchmark_results["dynamic"] = self.onnx_quantizer.benchmark_model(dynamic_path)
            if static_path:
                benchmark_results["static"] = self.onnx_quantizer.benchmark_model(static_path)
        
        end_time = time.time()
        
        return {
            "original_path": onnx_path,
            "dynamic_path": dynamic_path,
            "static_path": static_path,
            "benchmark": benchmark_results,
            "time_taken": end_time - start_time
        }
    
    def quantize_tflite(self, model_name: str, calib_dir: str, benchmark: bool) -> Dict[str, Any]:
        """TensorFlow Lite 量化"""
        start_time = time.time()
        
        # 转换模型
        tflite_path = self.tflite_quantizer.convert_insightface_to_tflite(model_name)
        
        # INT8 量化
        int8_path = None
        if calib_dir:
            calibration_images = self.tflite_quantizer.load_calibration_images(calib_dir, (1, 640, 640, 3))
            int8_path = self.tflite_quantizer.quantize_post_training_int8(tflite_path, calibration_images)
        
        # Float16 量化
        float16_path = self.tflite_quantizer.quantize_post_training_float16(tflite_path)
        
        # 动态量化
        dynamic_path = self.tflite_quantizer.quantize_dynamic_range(tflite_path)
        
        # 性能测试
        benchmark_results = {}
        if benchmark:
            benchmark_results["original"] = self.tflite_quantizer.benchmark_model(tflite_path)
            benchmark_results["float16"] = self.tflite_quantizer.benchmark_model(float16_path)
            benchmark_results["dynamic"] = self.tflite_quantizer.benchmark_model(dynamic_path)
            if int8_path:
                benchmark_results["int8"] = self.tflite_quantizer.benchmark_model(int8_path)
        
        # 模型大小对比
        size_comparison = {}
        size_comparison["float16"] = self.tflite_quantizer.compare_models(tflite_path, float16_path)
        size_comparison["dynamic"] = self.tflite_quantizer.compare_models(tflite_path, dynamic_path)
        if int8_path:
            size_comparison["int8"] = self.tflite_quantizer.compare_models(tflite_path, int8_path)
        
        end_time = time.time()
        
        return {
            "original_path": tflite_path,
            "int8_path": int8_path,
            "float16_path": float16_path,
            "dynamic_path": dynamic_path,
            "benchmark": benchmark_results,
            "size_comparison": size_comparison,
            "time_taken": end_time - start_time
        }
    
    def quantize_openvino(self, model_name: str, calib_dir: str, benchmark: bool) -> Dict[str, Any]:
        """OpenVINO 量化"""
        start_time = time.time()
        
        # 转换模型
        openvino_path = self.openvino_quantizer.convert_insightface_to_openvino(model_name)
        
        if not openvino_path:
            raise Exception("模型转换失败")
        
        # INT8 量化
        int8_path = None
        if calib_dir:
            int8_path = self.openvino_quantizer.quantize_int8(openvino_path, calib_dir)
        
        # FP16 量化
        fp16_path = self.openvino_quantizer.quantize_fp16(openvino_path)
        
        # 性能测试
        benchmark_results = {}
        if benchmark:
            benchmark_results["original"] = self.openvino_quantizer.benchmark_model(openvino_path)
            benchmark_results["fp16"] = self.openvino_quantizer.benchmark_model(fp16_path)
            if int8_path:
                benchmark_results["int8"] = self.openvino_quantizer.benchmark_model(int8_path)
        
        # 模型大小对比
        size_comparison = {}
        size_comparison["fp16"] = self.openvino_quantizer.compare_models(openvino_path, fp16_path)
        if int8_path:
            size_comparison["int8"] = self.openvino_quantizer.compare_models(openvino_path, int8_path)
        
        end_time = time.time()
        
        return {
            "original_path": openvino_path,
            "int8_path": int8_path,
            "fp16_path": fp16_path,
            "benchmark": benchmark_results,
            "size_comparison": size_comparison,
            "time_taken": end_time - start_time
        }
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成量化报告"""
        report = []
        report.append("# InsightFace 量化报告")
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for format_name, format_results in results.items():
            if "error" in format_results:
                report.append(f"## {format_name.upper()} 量化")
                report.append(f"❌ 失败: {format_results['error']}")
                report.append("")
                continue
            
            report.append(f"## {format_name.upper()} 量化")
            report.append(f"⏱️ 耗时: {format_results.get('time_taken', 0):.2f} 秒")
            report.append("")
            
            # 模型路径
            report.append("### 模型文件")
            for key, value in format_results.items():
                if key.endswith("_path") and value:
                    report.append(f"- {key}: `{value}`")
            report.append("")
            
            # 性能对比
            if "benchmark" in format_results:
                report.append("### 性能对比")
                benchmark = format_results["benchmark"]
                if benchmark:
                    report.append("| 模型类型 | 平均时间 (s) | FPS |")
                    report.append("|---------|-------------|-----|")
                    
                    for model_type, stats in benchmark.items():
                        if isinstance(stats, dict) and "fps" in stats:
                            report.append(f"| {model_type} | {stats['mean_time']:.4f} | {stats['fps']:.2f} |")
                report.append("")
            
            # 大小对比
            if "size_comparison" in format_results:
                report.append("### 模型大小对比")
                size_comp = format_results["size_comparison"]
                if size_comp:
                    report.append("| 模型类型 | 原始大小 (MB) | 量化大小 (MB) | 压缩比 | 大小减少 |")
                    report.append("|---------|-------------|-------------|--------|----------|")
                    
                    for model_type, comp in size_comp.items():
                        if isinstance(comp, dict):
                            orig_mb = comp["original_size"] / 1024 / 1024
                            quant_mb = comp["quantized_size"] / 1024 / 1024
                            report.append(f"| {model_type} | {orig_mb:.2f} | {quant_mb:.2f} | {comp['compression_ratio']:.2f}x | {comp['size_reduction']:.1f}% |")
                report.append("")
        
        return "\n".join(report)
    
    def save_report(self, results: Dict[str, Any], output_path: str = None):
        """保存量化报告"""
        if output_path is None:
            output_path = self.output_dir / "quantization_report.md"
        
        report = self.generate_report(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"量化报告已保存到: {output_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='InsightFace 统一量化工具')
    parser.add_argument('--model_name', type=str, default='buffalo_l', 
                       help='InsightFace 模型名称')
    parser.add_argument('--formats', nargs='+', 
                       choices=['onnx', 'tflite', 'openvino'], 
                       default=['onnx', 'tflite'],
                       help='量化格式')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='输出目录')
    parser.add_argument('--no_calibration', action='store_true',
                       help='不创建校准图像')
    parser.add_argument('--no_benchmark', action='store_true',
                       help='不运行性能测试')
    parser.add_argument('--report', type=str, default='quantization_report.md',
                       help='报告文件名')
    
    args = parser.parse_args()
    
    # 创建统一量化器
    quantizer = UnifiedQuantizer(args.output_dir)
    
    try:
        # 执行量化
        logger.info("开始 InsightFace 统一量化...")
        results = quantizer.quantize_all(
            model_name=args.model_name,
            quantization_types=args.formats,
            create_calibration=not args.no_calibration,
            benchmark=not args.no_benchmark
        )
        
        # 生成并保存报告
        quantizer.save_report(results, args.report)
        
        # 打印摘要
        logger.info("=== 量化完成 ===")
        for format_name, format_results in results.items():
            if "error" in format_results:
                logger.error(f"{format_name.upper()}: 失败 - {format_results['error']}")
            else:
                logger.info(f"{format_name.upper()}: 成功 - 耗时 {format_results.get('time_taken', 0):.2f}s")
        
        logger.info(f"详细报告: {args.report}")
        
    except Exception as e:
        logger.error(f"量化过程失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
