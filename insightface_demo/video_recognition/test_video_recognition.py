"""
视频人物识别测试脚本
包含单元测试、集成测试、性能测试等
"""

import unittest
import cv2
import numpy as np
import os
import tempfile
import time
import json
from unittest.mock import Mock, patch
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_face_recognition import VideoFaceRecognizer
from real_time_recognition import RealTimeVideoRecognizer
from performance_optimizer import PerformanceOptimizer, OptimizationLevel


class TestVideoFaceRecognizer(unittest.TestCase):
    """视频人脸识别器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.recognizer = VideoFaceRecognizer()
        
        # 创建测试图像
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_image_path = tempfile.mktemp(suffix='.jpg')
        cv2.imwrite(self.test_image_path, self.test_image)
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.recognizer.face_recognizer)
        self.assertIsNotNone(self.recognizer.tracked_faces)
        self.assertEqual(self.recognizer.next_track_id, 1)
    
    def test_detect_faces_in_frame(self):
        """测试人脸检测"""
        faces = self.recognizer.detect_faces_in_frame(self.test_image)
        self.assertIsInstance(faces, list)
    
    def test_track_faces(self):
        """测试人脸跟踪"""
        # 模拟检测到的人脸
        detected_faces = [
            {
                'face_id': 0,
                'bbox': [100, 100, 200, 200],
                'embedding': np.random.rand(512),
                'confidence': 0.9
            }
        ]
        
        tracked_faces = self.recognizer.track_faces(detected_faces)
        
        self.assertEqual(len(tracked_faces), 1)
        self.assertIn('track_id', tracked_faces[0])
        self.assertIn('tracked', tracked_faces[0])
    
    def test_calculate_face_distance(self):
        """测试人脸距离计算"""
        face1 = {
            'bbox': [100, 100, 200, 200],
            'embedding': np.random.rand(512)
        }
        face2 = {
            'bbox': [110, 110, 210, 210],
            'embedding': np.random.rand(512)
        }
        
        distance = self.recognizer.calculate_face_distance(face1, face2)
        self.assertGreaterEqual(distance, 0)
    
    def test_process_frame(self):
        """测试帧处理"""
        processed_frame = self.recognizer.process_frame(self.test_image)
        
        self.assertIsInstance(processed_frame, np.ndarray)
        self.assertEqual(processed_frame.shape, self.test_image.shape)
    
    def test_stats_update(self):
        """测试统计信息更新"""
        initial_frames = self.recognizer.stats['total_frames']
        
        self.recognizer.process_frame(self.test_image)
        
        self.assertEqual(self.recognizer.stats['total_frames'], initial_frames + 1)


class TestRealTimeVideoRecognizer(unittest.TestCase):
    """实时视频识别器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.recognizer = RealTimeVideoRecognizer(
            max_queue_size=2,
            processing_threads=1
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.recognizer.face_recognizer)
        self.assertIsNotNone(self.recognizer.frame_queue)
        self.assertIsNotNone(self.recognizer.result_queue)
        self.assertFalse(self.recognizer.running)
    
    def test_set_callbacks(self):
        """测试回调设置"""
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()
        
        self.recognizer.set_callbacks(
            on_face_detected=callback1,
            on_face_recognized=callback2,
            on_frame_processed=callback3
        )
        
        self.assertEqual(self.recognizer.on_face_detected, callback1)
        self.assertEqual(self.recognizer.on_face_recognized, callback2)
        self.assertEqual(self.recognizer.on_frame_processed, callback3)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        stats = self.recognizer.get_stats()
        
        self.assertIn('frames_processed', stats)
        self.assertIn('faces_detected', stats)
        self.assertIn('faces_recognized', stats)
        self.assertIn('processing_fps', stats)
        self.assertIn('queue_size', stats)
    
    def test_save_recognition_log(self):
        """测试保存识别日志"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            log_path = f.name
        
        try:
            self.recognizer.save_recognition_log(log_path)
            
            self.assertTrue(os.path.exists(log_path))
            
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            self.assertIn('timestamp', log_data)
            self.assertIn('stats', log_data)
            self.assertIn('settings', log_data)
        
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)


class TestPerformanceOptimizer(unittest.TestCase):
    """性能优化器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.optimizer = PerformanceOptimizer()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.optimizer.monitor)
        self.assertIsNotNone(self.optimizer.model_optimizer)
        self.assertIsNotNone(self.optimizer.memory_manager)
        self.assertIsNotNone(self.optimizer.system_specs)
    
    def test_get_system_specs(self):
        """测试获取系统规格"""
        specs = self.optimizer.system_specs
        
        self.assertIn('cpu_count', specs)
        self.assertIn('memory_gb', specs)
        self.assertIn('gpu_available', specs)
        self.assertIn('platform', specs)
        
        self.assertGreater(specs['cpu_count'], 0)
        self.assertGreater(specs['memory_gb'], 0)
        self.assertIsInstance(specs['gpu_available'], bool)
    
    def test_apply_optimization(self):
        """测试应用优化"""
        for level in OptimizationLevel:
            config = self.optimizer.apply_optimization(level)
            
            self.assertIn('det_size', config)
            self.assertIn('batch_size', config)
            self.assertIn('max_faces', config)
            self.assertIn('enable_gpu', config)
            self.assertIn('thread_count', config)
    
    def test_get_optimization_recommendations(self):
        """测试获取优化建议"""
        recommendations = self.optimizer.get_optimization_recommendations()
        
        self.assertIsInstance(recommendations, list)
        
        for rec in recommendations:
            self.assertIn('type', rec)
            self.assertIn('priority', rec)
            self.assertIn('title', rec)
            self.assertIn('description', rec)
            self.assertIn('action', rec)
    
    def test_get_performance_report(self):
        """测试获取性能报告"""
        report = self.optimizer.get_performance_report()
        
        self.assertIn('performance_summary', report)
        self.assertIn('memory_info', report)
        self.assertIn('system_specs', report)
        self.assertIn('recommendations', report)
        self.assertIn('timestamp', report)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.recognizer = VideoFaceRecognizer()
        
        # 创建测试视频
        self.test_video_path = tempfile.mktemp(suffix='.mp4')
        self.create_test_video()
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)
    
    def create_test_video(self):
        """创建测试视频"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.test_video_path, fourcc, 30.0, (640, 480))
        
        for i in range(30):  # 1秒视频
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
    
    def test_video_processing(self):
        """测试视频处理"""
        success = self.recognizer.process_video_file(
            self.test_video_path, 
            display=False
        )
        
        self.assertTrue(success)
        self.assertGreater(self.recognizer.stats['total_frames'], 0)
    
    def test_camera_processing(self):
        """测试摄像头处理（模拟）"""
        # 模拟摄像头处理
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            mock_cap.get.side_effect = lambda x: {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480, cv2.CAP_PROP_FRAME_COUNT: 30}[x]
            mock_capture.return_value = mock_cap
            
            success = self.recognizer.process_camera(display=False)
            self.assertTrue(success)


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def setUp(self):
        """测试前准备"""
        self.recognizer = VideoFaceRecognizer()
        self.optimizer = PerformanceOptimizer()
    
    def test_processing_speed(self):
        """测试处理速度"""
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试处理时间
        start_time = time.time()
        
        for _ in range(10):
            self.recognizer.process_frame(test_image)
        
        end_time = time.time()
        processing_time = (end_time - start_time) / 10
        
        # 处理时间应该小于0.1秒
        self.assertLess(processing_time, 0.1)
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # 处理一些帧
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        for _ in range(50):
            self.recognizer.process_frame(test_image)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
        
        # 内存增长应该小于100MB
        self.assertLess(memory_increase, 100)
    
    def test_concurrent_processing(self):
        """测试并发处理"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def process_frames():
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            start_time = time.time()
            
            for _ in range(10):
                self.recognizer.process_frame(test_image)
            
            end_time = time.time()
            results.put(end_time - start_time)
        
        # 启动多个线程
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=process_frames)
            thread.start()
            threads.append(thread)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查结果
        self.assertEqual(results.qsize(), 3)
        
        # 所有处理时间都应该合理
        while not results.empty():
            processing_time = results.get()
            self.assertLess(processing_time, 1.0)


def run_benchmark():
    """运行基准测试"""
    print("=== 视频人物识别基准测试 ===")
    
    # 创建识别器
    recognizer = VideoFaceRecognizer()
    
    # 测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 预热
    print("预热中...")
    for _ in range(10):
        recognizer.process_frame(test_image)
    
    # 性能测试
    print("性能测试中...")
    num_frames = 100
    start_time = time.time()
    
    for i in range(num_frames):
        recognizer.process_frame(test_image)
        if (i + 1) % 20 == 0:
            print(f"已处理 {i + 1}/{num_frames} 帧")
    
    end_time = time.time()
    
    # 计算性能指标
    total_time = end_time - start_time
    fps = num_frames / total_time
    avg_processing_time = total_time / num_frames
    
    print(f"\n性能结果:")
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"平均FPS: {fps:.2f}")
    print(f"平均处理时间: {avg_processing_time:.3f} 秒/帧")
    print(f"检测到人脸: {recognizer.stats['total_faces_detected']}")
    print(f"识别人脸: {recognizer.stats['total_faces_recognized']}")
    
    # 内存使用
    import psutil
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"内存使用: {memory_usage:.1f} MB")


def run_integration_test():
    """运行集成测试"""
    print("=== 集成测试 ===")
    
    # 创建测试视频
    test_video_path = tempfile.mktemp(suffix='.mp4')
    
    try:
        # 创建测试视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640, 480))
        
        print("创建测试视频...")
        for i in range(90):  # 3秒视频
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # 测试视频处理
        print("测试视频处理...")
        recognizer = VideoFaceRecognizer()
        
        start_time = time.time()
        success = recognizer.process_video_file(test_video_path, display=False)
        end_time = time.time()
        
        if success:
            print(f"✓ 视频处理成功")
            print(f"  处理时间: {end_time - start_time:.2f} 秒")
            print(f"  处理帧数: {recognizer.stats['total_frames']}")
            print(f"  检测人脸: {recognizer.stats['total_faces_detected']}")
        else:
            print("✗ 视频处理失败")
        
        # 测试实时识别
        print("\n测试实时识别...")
        real_time_recognizer = RealTimeVideoRecognizer(
            max_queue_size=3,
            processing_threads=2
        )
        
        # 模拟处理
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        for _ in range(50):
            real_time_recognizer.face_recognizer.process_frame(test_image)
        end_time = time.time()
        
        print(f"✓ 实时识别测试完成")
        print(f"  处理时间: {end_time - start_time:.2f} 秒")
        print(f"  平均FPS: {50 / (end_time - start_time):.2f}")
        
    finally:
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


if __name__ == "__main__":
    # 运行单元测试
    print("运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行基准测试
    print("\n" + "="*50)
    run_benchmark()
    
    # 运行集成测试
    print("\n" + "="*50)
    run_integration_test()
