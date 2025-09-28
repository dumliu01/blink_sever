"""
性能优化和配置管理工具
包含模型优化、内存管理、性能监控等功能
"""

import cv2
import numpy as np
import time
import psutil
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime
import gc
import multiprocessing
from dataclasses import dataclass
from enum import Enum


class OptimizationLevel(Enum):
    """优化级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class PerformanceMetrics:
    """性能指标"""
    fps: float
    memory_usage: float
    cpu_usage: float
    processing_time: float
    queue_size: int
    frame_drop_rate: float
    recognition_accuracy: float
    timestamp: datetime


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, window_size: int = 60):
        """
        初始化性能监控器
        
        Args:
            window_size: 监控窗口大小（秒）
        """
        self.window_size = window_size
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
        
        # 性能阈值
        self.thresholds = {
            'fps_min': 15.0,
            'memory_max': 80.0,  # 百分比
            'cpu_max': 80.0,     # 百分比
            'processing_time_max': 0.1,  # 秒
            'queue_size_max': 10,
            'frame_drop_rate_max': 0.1   # 10%
        }
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 收集性能指标
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 保持历史记录在窗口大小内
                current_time = time.time()
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if (current_time - m.timestamp.timestamp()) <= self.window_size
                ]
                
                time.sleep(1)  # 每秒更新一次
                
            except Exception as e:
                print(f"性能监控错误: {e}")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        # CPU使用率
        cpu_usage = psutil.cpu_percent()
        
        # 内存使用率
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # 创建基础指标
        metrics = PerformanceMetrics(
            fps=0.0,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            processing_time=0.0,
            queue_size=0,
            frame_drop_rate=0.0,
            recognition_accuracy=0.0,
            timestamp=datetime.now()
        )
        
        return metrics
    
    def update_metrics(self, fps: float, processing_time: float, 
                      queue_size: int, frame_drop_rate: float, 
                      recognition_accuracy: float):
        """更新性能指标"""
        if self.metrics_history:
            latest = self.metrics_history[-1]
            latest.fps = fps
            latest.processing_time = processing_time
            latest.queue_size = queue_size
            latest.frame_drop_rate = frame_drop_rate
            latest.recognition_accuracy = recognition_accuracy
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # 最近10个指标
        
        return {
            'avg_fps': np.mean([m.fps for m in recent_metrics]),
            'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'avg_cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
            'avg_processing_time': np.mean([m.processing_time for m in recent_metrics]),
            'avg_queue_size': np.mean([m.queue_size for m in recent_metrics]),
            'avg_frame_drop_rate': np.mean([m.frame_drop_rate for m in recent_metrics]),
            'avg_recognition_accuracy': np.mean([m.recognition_accuracy for m in recent_metrics]),
            'performance_score': self._calculate_performance_score(recent_metrics)
        }
    
    def _calculate_performance_score(self, metrics: List[PerformanceMetrics]) -> float:
        """计算性能评分 (0-100)"""
        if not metrics:
            return 0.0
        
        scores = []
        
        # FPS评分
        avg_fps = np.mean([m.fps for m in metrics])
        fps_score = min(100, (avg_fps / 30) * 100)  # 30fps为满分
        scores.append(fps_score)
        
        # 内存使用评分
        avg_memory = np.mean([m.memory_usage for m in metrics])
        memory_score = max(0, 100 - avg_memory)  # 内存使用越少分数越高
        scores.append(memory_score)
        
        # CPU使用评分
        avg_cpu = np.mean([m.cpu_usage for m in metrics])
        cpu_score = max(0, 100 - avg_cpu)  # CPU使用越少分数越高
        scores.append(cpu_score)
        
        # 处理时间评分
        avg_processing_time = np.mean([m.processing_time for m in metrics])
        processing_score = max(0, 100 - (avg_processing_time * 1000))  # 转换为毫秒
        scores.append(processing_score)
        
        # 识别准确率评分
        avg_accuracy = np.mean([m.recognition_accuracy for m in metrics])
        accuracy_score = avg_accuracy * 100
        scores.append(accuracy_score)
        
        return np.mean(scores)
    
    def check_performance_issues(self) -> List[str]:
        """检查性能问题"""
        issues = []
        
        if not self.metrics_history:
            return issues
        
        recent_metrics = self.metrics_history[-5:]  # 最近5个指标
        
        # 检查FPS
        avg_fps = np.mean([m.fps for m in recent_metrics])
        if avg_fps < self.thresholds['fps_min']:
            issues.append(f"FPS过低: {avg_fps:.1f} < {self.thresholds['fps_min']}")
        
        # 检查内存使用
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        if avg_memory > self.thresholds['memory_max']:
            issues.append(f"内存使用过高: {avg_memory:.1f}% > {self.thresholds['memory_max']}%")
        
        # 检查CPU使用
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        if avg_cpu > self.thresholds['cpu_max']:
            issues.append(f"CPU使用过高: {avg_cpu:.1f}% > {self.thresholds['cpu_max']}%")
        
        # 检查处理时间
        avg_processing_time = np.mean([m.processing_time for m in recent_metrics])
        if avg_processing_time > self.thresholds['processing_time_max']:
            issues.append(f"处理时间过长: {avg_processing_time:.3f}s > {self.thresholds['processing_time_max']}s")
        
        # 检查队列大小
        avg_queue_size = np.mean([m.queue_size for m in recent_metrics])
        if avg_queue_size > self.thresholds['queue_size_max']:
            issues.append(f"队列积压: {avg_queue_size:.1f} > {self.thresholds['queue_size_max']}")
        
        # 检查丢帧率
        avg_frame_drop_rate = np.mean([m.frame_drop_rate for m in recent_metrics])
        if avg_frame_drop_rate > self.thresholds['frame_drop_rate_max']:
            issues.append(f"丢帧率过高: {avg_frame_drop_rate:.1%} > {self.thresholds['frame_drop_rate_max']:.1%}")
        
        return issues


class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self):
        self.optimization_configs = {
            OptimizationLevel.LOW: {
                'det_size': (320, 320),
                'batch_size': 1,
                'max_faces': 5,
                'enable_gpu': False,
                'thread_count': 1
            },
            OptimizationLevel.MEDIUM: {
                'det_size': (640, 640),
                'batch_size': 2,
                'max_faces': 10,
                'enable_gpu': True,
                'thread_count': 2
            },
            OptimizationLevel.HIGH: {
                'det_size': (640, 640),
                'batch_size': 4,
                'max_faces': 20,
                'enable_gpu': True,
                'thread_count': 4
            },
            OptimizationLevel.ULTRA: {
                'det_size': (1024, 1024),
                'batch_size': 8,
                'max_faces': 50,
                'enable_gpu': True,
                'thread_count': 8
            }
        }
    
    def get_optimized_config(self, level: OptimizationLevel, 
                           system_specs: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取优化配置
        
        Args:
            level: 优化级别
            system_specs: 系统规格
            
        Returns:
            优化配置
        """
        base_config = self.optimization_configs[level].copy()
        
        # 根据系统规格调整
        cpu_count = system_specs.get('cpu_count', multiprocessing.cpu_count())
        memory_gb = system_specs.get('memory_gb', psutil.virtual_memory().total / (1024**3))
        gpu_available = system_specs.get('gpu_available', False)
        
        # 调整线程数
        max_threads = min(cpu_count, base_config['thread_count'])
        base_config['thread_count'] = max(1, max_threads)
        
        # 调整批处理大小
        if memory_gb < 4:
            base_config['batch_size'] = 1
        elif memory_gb < 8:
            base_config['batch_size'] = min(2, base_config['batch_size'])
        
        # 调整检测尺寸
        if memory_gb < 4:
            base_config['det_size'] = (320, 320)
        elif memory_gb < 8:
            base_config['det_size'] = (480, 480)
        
        # 禁用GPU如果不可用
        if not gpu_available:
            base_config['enable_gpu'] = False
        
        return base_config
    
    def optimize_model_loading(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化模型加载
        
        Args:
            model_name: 模型名称
            config: 配置
            
        Returns:
            优化后的配置
        """
        optimized_config = config.copy()
        
        # 根据模型大小调整配置
        if 'buffalo_s' in model_name:
            # 小模型，可以使用更高分辨率
            optimized_config['det_size'] = (640, 640)
        elif 'buffalo_m' in model_name:
            # 中等模型
            optimized_config['det_size'] = (480, 480)
        elif 'buffalo_l' in model_name:
            # 大模型，降低分辨率
            optimized_config['det_size'] = (320, 320)
        
        return optimized_config


class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_memory_usage: float = 80.0):
        """
        初始化内存管理器
        
        Args:
            max_memory_usage: 最大内存使用率（百分比）
        """
        self.max_memory_usage = max_memory_usage
        self.memory_cleanup_interval = 30  # 秒
        self.last_cleanup = time.time()
    
    def check_memory_usage(self) -> bool:
        """检查内存使用情况"""
        memory_usage = psutil.virtual_memory().percent
        return memory_usage < self.max_memory_usage
    
    def cleanup_memory(self):
        """清理内存"""
        # 强制垃圾回收
        gc.collect()
        
        # 更新清理时间
        self.last_cleanup = time.time()
    
    def should_cleanup(self) -> bool:
        """判断是否需要清理内存"""
        current_time = time.time()
        return (current_time - self.last_cleanup) > self.memory_cleanup_interval
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percentage': memory.percent,
            'free': memory.free
        }


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.model_optimizer = ModelOptimizer()
        self.memory_manager = MemoryManager()
        
        # 优化建议
        self.optimization_suggestions = []
        
        # 系统规格
        self.system_specs = self._get_system_specs()
    
    def _get_system_specs(self) -> Dict[str, Any]:
        """获取系统规格"""
        return {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': self._check_gpu_availability(),
            'platform': os.name
        }
    
    def _check_gpu_availability(self) -> bool:
        """检查GPU可用性"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def start_optimization(self):
        """开始优化"""
        self.monitor.start_monitoring()
        print("性能优化已启动")
    
    def stop_optimization(self):
        """停止优化"""
        self.monitor.stop_monitoring()
        print("性能优化已停止")
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """获取优化建议"""
        recommendations = []
        
        # 检查性能问题
        issues = self.monitor.check_performance_issues()
        
        for issue in issues:
            if "FPS过低" in issue:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'high',
                    'title': '提高FPS',
                    'description': '降低检测分辨率或减少处理线程数',
                    'action': 'reduce_detection_size'
                })
            
            elif "内存使用过高" in issue:
                recommendations.append({
                    'type': 'memory',
                    'priority': 'high',
                    'title': '降低内存使用',
                    'description': '减少批处理大小或启用内存清理',
                    'action': 'reduce_batch_size'
                })
            
            elif "CPU使用过高" in issue:
                recommendations.append({
                    'type': 'cpu',
                    'priority': 'medium',
                    'title': '降低CPU使用',
                    'description': '减少处理线程数或使用更小的模型',
                    'action': 'reduce_threads'
                })
        
        # 根据系统规格提供建议
        if self.system_specs['memory_gb'] < 4:
            recommendations.append({
                'type': 'system',
                'priority': 'medium',
                'title': '系统内存不足',
                'description': '建议使用小模型和低分辨率检测',
                'action': 'use_small_model'
            })
        
        if not self.system_specs['gpu_available']:
            recommendations.append({
                'type': 'system',
                'priority': 'low',
                'title': 'GPU不可用',
                'description': '建议安装CUDA以提升性能',
                'action': 'install_cuda'
            })
        
        return recommendations
    
    def apply_optimization(self, optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """应用优化"""
        config = self.model_optimizer.get_optimized_config(
            optimization_level, self.system_specs
        )
        
        # 优化模型加载
        config = self.model_optimizer.optimize_model_loading('buffalo_l', config)
        
        return config
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        summary = self.monitor.get_performance_summary()
        memory_info = self.memory_manager.get_memory_info()
        recommendations = self.get_optimization_recommendations()
        
        return {
            'performance_summary': summary,
            'memory_info': memory_info,
            'system_specs': self.system_specs,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_performance_report(self, filepath: str):
        """保存性能报告"""
        report = self.get_performance_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"性能报告已保存到: {filepath}")


def demo_performance_optimization():
    """性能优化演示"""
    print("=== 性能优化演示 ===")
    
    # 创建性能优化器
    optimizer = PerformanceOptimizer()
    
    # 显示系统规格
    print("\n1. 系统规格:")
    specs = optimizer.system_specs
    print(f"  CPU核心数: {specs['cpu_count']}")
    print(f"  内存大小: {specs['memory_gb']:.1f} GB")
    print(f"  GPU可用: {specs['gpu_available']}")
    print(f"  平台: {specs['platform']}")
    
    # 获取优化建议
    print("\n2. 优化建议:")
    recommendations = optimizer.get_optimization_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. [{rec['priority'].upper()}] {rec['title']}")
        print(f"     {rec['description']}")
    
    # 应用优化
    print("\n3. 应用优化配置:")
    for level in OptimizationLevel:
        config = optimizer.apply_optimization(level)
        print(f"  {level.value.upper()}:")
        for key, value in config.items():
            print(f"    {key}: {value}")
    
    # 启动优化监控
    print("\n4. 启动性能监控...")
    optimizer.start_optimization()
    
    try:
        # 模拟运行一段时间
        print("监控中... (按Ctrl+C停止)")
        time.sleep(10)
        
        # 生成性能报告
        print("\n5. 生成性能报告...")
        report = optimizer.get_performance_report()
        
        print(f"性能评分: {report['performance_summary'].get('performance_score', 0):.1f}")
        print(f"平均FPS: {report['performance_summary'].get('avg_fps', 0):.1f}")
        print(f"内存使用: {report['memory_info']['percentage']:.1f}%")
        print(f"CPU使用: {report['performance_summary'].get('avg_cpu_usage', 0):.1f}%")
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"performance_report_{timestamp}.json"
        optimizer.save_performance_report(report_file)
        
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        # 停止优化
        optimizer.stop_optimization()
        print("性能优化已停止")


if __name__ == "__main__":
    demo_performance_optimization()
