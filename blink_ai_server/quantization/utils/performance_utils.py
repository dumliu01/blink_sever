"""
性能测试工具类
提供模型性能测试和基准测试功能
"""

import time
import psutil
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Optional
import logging
import threading
import queue

logger = logging.getLogger(__name__)

class PerformanceUtils:
    """性能测试工具类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def benchmark_model(self, model_path: str, test_inputs: List[np.ndarray], 
                       warmup_runs: int = 10, test_runs: int = 100) -> Dict[str, Any]:
        """
        对模型进行性能基准测试
        
        Args:
            model_path: 模型文件路径
            test_inputs: 测试输入数据
            warmup_runs: 预热运行次数
            test_runs: 测试运行次数
            
        Returns:
            Dict: 性能测试结果
        """
        try:
            # 创建推理会话
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            # 预热
            self.logger.info(f"开始预热，运行 {warmup_runs} 次...")
            for _ in range(warmup_runs):
                for test_input in test_inputs:
                    _ = session.run(None, {input_name: test_input})
            
            # 性能测试
            self.logger.info(f"开始性能测试，运行 {test_runs} 次...")
            times = []
            memory_usage = []
            
            for i in range(test_runs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                for test_input in test_inputs:
                    _ = session.run(None, {input_name: test_input})
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                times.append(end_time - start_time)
                memory_usage.append(end_memory - start_memory)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"已完成 {i + 1}/{test_runs} 次测试")
            
            # 计算统计信息
            times = np.array(times)
            memory_usage = np.array(memory_usage)
            
            results = {
                "model_path": model_path,
                "warmup_runs": warmup_runs,
                "test_runs": test_runs,
                "total_time": float(np.sum(times)),
                "avg_time": float(np.mean(times)),
                "min_time": float(np.min(times)),
                "max_time": float(np.max(times)),
                "std_time": float(np.std(times)),
                "fps": float(len(test_inputs) * test_runs / np.sum(times)),
                "avg_memory_mb": float(np.mean(memory_usage) / (1024 * 1024)),
                "max_memory_mb": float(np.max(memory_usage) / (1024 * 1024)),
                "success": True
            }
            
            self.logger.info(f"性能测试完成: {results['fps']:.2f} FPS")
            return results
            
        except Exception as e:
            self.logger.error(f"性能测试失败: {e}")
            return {"success": False, "error": str(e)}
    
    def compare_performance(self, models: List[Dict[str, str]], 
                           test_inputs: List[np.ndarray]) -> Dict[str, Any]:
        """
        比较多个模型的性能
        
        Args:
            models: 模型列表，每个元素包含 name 和 path
            test_inputs: 测试输入数据
            
        Returns:
            Dict: 性能比较结果
        """
        results = {
            "models": [],
            "comparison": {},
            "success": True
        }
        
        try:
            # 测试每个模型
            for model_info in models:
                self.logger.info(f"测试模型: {model_info['name']}")
                perf_result = self.benchmark_model(
                    model_info['path'], 
                    test_inputs
                )
                
                if perf_result.get("success", False):
                    results["models"].append({
                        "name": model_info['name'],
                        "path": model_info['path'],
                        **perf_result
                    })
                else:
                    self.logger.error(f"模型 {model_info['name']} 测试失败")
            
            # 性能比较
            if len(results["models"]) > 1:
                baseline = results["models"][0]
                for model in results["models"][1:]:
                    speedup = baseline["avg_time"] / model["avg_time"]
                    size_ratio = baseline.get("file_size_mb", 0) / model.get("file_size_mb", 1)
                    
                    results["comparison"][model["name"]] = {
                        "speedup": float(speedup),
                        "size_ratio": float(size_ratio),
                        "fps_improvement": model["fps"] - baseline["fps"]
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"性能比较失败: {e}")
            return {"success": False, "error": str(e)}
    
    def memory_profiling(self, model_path: str, test_inputs: List[np.ndarray]) -> Dict[str, Any]:
        """
        内存使用分析
        
        Args:
            model_path: 模型文件路径
            test_inputs: 测试输入数据
            
        Returns:
            Dict: 内存分析结果
        """
        try:
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            # 记录内存使用
            memory_snapshots = []
            
            for i, test_input in enumerate(test_inputs):
                # 推理前内存
                before_memory = psutil.Process().memory_info().rss
                
                # 执行推理
                _ = session.run(None, {input_name: test_input})
                
                # 推理后内存
                after_memory = psutil.Process().memory_info().rss
                
                memory_snapshots.append({
                    "input_index": i,
                    "before_mb": before_memory / (1024 * 1024),
                    "after_mb": after_memory / (1024 * 1024),
                    "delta_mb": (after_memory - before_memory) / (1024 * 1024)
                })
            
            # 计算统计信息
            deltas = [snapshot["delta_mb"] for snapshot in memory_snapshots]
            
            return {
                "model_path": model_path,
                "snapshots": memory_snapshots,
                "avg_delta_mb": float(np.mean(deltas)),
                "max_delta_mb": float(np.max(deltas)),
                "min_delta_mb": float(np.min(deltas)),
                "std_delta_mb": float(np.std(deltas)),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"内存分析失败: {e}")
            return {"success": False, "error": str(e)}
    
    def stress_test(self, model_path: str, test_inputs: List[np.ndarray], 
                   duration_seconds: int = 60) -> Dict[str, Any]:
        """
        压力测试
        
        Args:
            model_path: 模型文件路径
            test_inputs: 测试输入数据
            duration_seconds: 测试持续时间（秒）
            
        Returns:
            Dict: 压力测试结果
        """
        try:
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            run_count = 0
            error_count = 0
            times = []
            
            self.logger.info(f"开始压力测试，持续 {duration_seconds} 秒...")
            
            while time.time() < end_time:
                try:
                    start_run = time.time()
                    
                    for test_input in test_inputs:
                        _ = session.run(None, {input_name: test_input})
                    
                    end_run = time.time()
                    times.append(end_run - start_run)
                    run_count += 1
                    
                except Exception as e:
                    error_count += 1
                    self.logger.warning(f"压力测试中发生错误: {e}")
            
            actual_duration = time.time() - start_time
            times = np.array(times)
            
            return {
                "model_path": model_path,
                "duration_seconds": actual_duration,
                "total_runs": run_count,
                "error_count": error_count,
                "success_rate": (run_count - error_count) / max(run_count, 1),
                "avg_time": float(np.mean(times)) if len(times) > 0 else 0,
                "min_time": float(np.min(times)) if len(times) > 0 else 0,
                "max_time": float(np.max(times)) if len(times) > 0 else 0,
                "fps": run_count * len(test_inputs) / actual_duration,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"压力测试失败: {e}")
            return {"success": False, "error": str(e)}
