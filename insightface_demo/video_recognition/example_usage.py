#!/usr/bin/env python3
"""
视频人物识别使用示例
展示各种使用场景和最佳实践
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from video_face_recognition import VideoFaceRecognizer
from real_time_recognition import RealTimeVideoRecognizer
from performance_optimizer import PerformanceOptimizer, OptimizationLevel


def example_basic_recognition():
    """基础识别示例"""
    print("=== 基础识别示例 ===")
    
    # 创建识别器
    recognizer = VideoFaceRecognizer(
        similarity_threshold=0.6,
        tracking_threshold=0.5,
        max_tracking_distance=100.0
    )
    
    # 注册人员
    print("1. 注册人员...")
    test_images = {
        "person1": "../test_images/person1_1.jpg",
        "person2": "../test_images/person2_1.jpg",
        "person2_2": "../test_images/person2_2.jpg",
        "person2_3": "../test_images/person2_3.jpg"
    }
    
    for person_id, img_path in test_images.items():
        if os.path.exists(img_path):
            person_name = f"人员{person_id}"
            success = recognizer.face_recognizer.register_person(person_id, person_name, img_path)
            if success:
                print(f"✓ 注册成功: {person_name}")
            else:
                print(f"✗ 注册失败: {person_name}")
    
    # 显示已注册人员
    print("\n2. 已注册人员:")
    persons = recognizer.face_recognizer.get_all_persons()
    for person in persons:
        print(f"  {person['person_name']} (ID: {person['person_id']}) - {person['image_count']}张图像")
    
    # 处理视频文件
    print("\n3. 处理视频文件...")
    video_path = "test_video.mp4"
    if os.path.exists(video_path):
        success = recognizer.process_video_file(video_path, "output_video.mp4", display=False)
        if success:
            print("✓ 视频处理完成")
        else:
            print("✗ 视频处理失败")
    else:
        print("测试视频文件不存在，跳过视频处理")
    
    # 显示统计信息
    print("\n4. 统计信息:")
    stats = recognizer.get_stats_summary()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_realtime_recognition():
    """实时识别示例"""
    print("\n=== 实时识别示例 ===")
    
    # 创建实时识别器
    recognizer = RealTimeVideoRecognizer(
        similarity_threshold=0.6,
        max_queue_size=5,
        processing_threads=2
    )
    
    # 设置回调函数
    def on_face_recognized(face_info, frame_count):
        person_name = face_info.get('person_name', 'Unknown')
        similarity = face_info.get('similarity', 0.0)
        track_id = face_info.get('track_id', 0)
        print(f"识别到: {person_name} (相似度: {similarity:.3f}, 跟踪ID: {track_id})")
    
    def on_face_detected(face_info, frame_count):
        track_id = face_info.get('track_id', 0)
        confidence = face_info.get('confidence', 0.0)
        print(f"检测到未知人脸 (跟踪ID: {track_id}, 置信度: {confidence:.3f})")
    
    def on_frame_processed(frame, frame_count, processing_time):
        if frame_count % 30 == 0:  # 每30帧打印一次
            print(f"已处理 {frame_count} 帧，处理时间: {processing_time:.3f}s")
    
    recognizer.set_callbacks(
        on_face_detected=on_face_detected,
        on_face_recognized=on_face_recognized,
        on_frame_processed=on_frame_processed
    )
    
    # 注册测试人员
    print("1. 注册测试人员...")
    test_images = {
        "person1": "../test_images/person1_1.jpg",
        "person2": "../test_images/person2_1.jpg"
    }
    
    for person_id, img_path in test_images.items():
        if os.path.exists(img_path):
            person_name = f"人员{person_id}"
            recognizer.face_recognizer.face_recognizer.register_person(person_id, person_name, img_path)
            print(f"✓ 注册: {person_name}")
    
    # 启动识别
    print("\n2. 启动实时识别...")
    print("按Ctrl+C停止识别")
    
    try:
        recognizer.start(camera_id=0)
        
        # 运行一段时间
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        recognizer.stop()
        
        # 显示统计信息
        print("\n3. 统计信息:")
        stats = recognizer.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")


def example_performance_optimization():
    """性能优化示例"""
    print("\n=== 性能优化示例 ===")
    
    # 创建性能优化器
    optimizer = PerformanceOptimizer()
    
    # 显示系统规格
    print("1. 系统规格:")
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
    
    # 应用不同级别的优化
    print("\n3. 优化配置:")
    for level in OptimizationLevel:
        config = optimizer.apply_optimization(level)
        print(f"  {level.value.upper()}:")
        for key, value in config.items():
            print(f"    {key}: {value}")
    
    # 启动性能监控
    print("\n4. 性能监控:")
    optimizer.start_optimization()
    
    try:
        # 模拟一些工作
        print("监控中...")
        time.sleep(5)
        
        # 获取性能报告
        report = optimizer.get_performance_report()
        print(f"性能评分: {report['performance_summary'].get('performance_score', 0):.1f}")
        print(f"平均FPS: {report['performance_summary'].get('avg_fps', 0):.1f}")
        print(f"内存使用: {report['memory_info']['percentage']:.1f}%")
        
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        optimizer.stop_optimization()


def example_custom_callback():
    """自定义回调示例"""
    print("\n=== 自定义回调示例 ===")
    
    class CustomRecognizer(RealTimeVideoRecognizer):
        """自定义识别器"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recognition_count = 0
            self.detection_count = 0
        
        def on_face_recognized(self, face_info, frame_count):
            """自定义识别回调"""
            self.recognition_count += 1
            person_name = face_info.get('person_name', 'Unknown')
            similarity = face_info.get('similarity', 0.0)
            
            print(f"[识别 #{self.recognition_count}] {person_name} (相似度: {similarity:.3f})")
            
            # 可以在这里添加自定义逻辑
            if similarity > 0.8:
                print(f"  -> 高置信度识别: {person_name}")
        
        def on_face_detected(self, face_info, frame_count):
            """自定义检测回调"""
            self.detection_count += 1
            track_id = face_info.get('track_id', 0)
            confidence = face_info.get('confidence', 0.0)
            
            print(f"[检测 #{self.detection_count}] 未知人脸 (跟踪ID: {track_id}, 置信度: {confidence:.3f})")
        
        def get_custom_stats(self):
            """获取自定义统计"""
            base_stats = self.get_stats()
            base_stats['custom_recognition_count'] = self.recognition_count
            base_stats['custom_detection_count'] = self.detection_count
            return base_stats
    
    # 使用自定义识别器
    custom_recognizer = CustomRecognizer()
    
    # 注册测试人员
    test_images = {
        "person1": "../test_images/person1_1.jpg",
        "person2": "../test_images/person2_1.jpg"
    }
    
    for person_id, img_path in test_images.items():
        if os.path.exists(img_path):
            person_name = f"人员{person_id}"
            custom_recognizer.face_recognizer.face_recognizer.register_person(person_id, person_name, img_path)
    
    print("自定义识别器已创建，包含自定义回调函数")


def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    # 创建识别器
    recognizer = VideoFaceRecognizer()
    
    # 注册人员
    test_images = {
        "person1": "../test_images/person1_1.jpg",
        "person2": "../test_images/person2_1.jpg"
    }
    
    for person_id, img_path in test_images.items():
        if os.path.exists(img_path):
            person_name = f"人员{person_id}"
            recognizer.face_recognizer.register_person(person_id, person_name, img_path)
    
    # 批量处理多个视频文件
    video_files = [
        "video1.mp4",
        "video2.mp4",
        "video3.mp4"
    ]
    
    results = []
    
    for video_file in video_files:
        if os.path.exists(video_file):
            print(f"处理视频: {video_file}")
            
            # 重置统计
            recognizer.reset_stats()
            
            # 处理视频
            start_time = time.time()
            success = recognizer.process_video_file(video_file, display=False)
            end_time = time.time()
            
            if success:
                stats = recognizer.get_stats_summary()
                results.append({
                    'video_file': video_file,
                    'success': True,
                    'processing_time': end_time - start_time,
                    'frames_processed': stats['total_frames'],
                    'faces_detected': stats['total_faces_detected'],
                    'faces_recognized': stats['total_faces_recognized']
                })
                print(f"✓ 处理完成: {end_time - start_time:.2f}秒")
            else:
                results.append({
                    'video_file': video_file,
                    'success': False,
                    'error': '处理失败'
                })
                print(f"✗ 处理失败")
        else:
            print(f"视频文件不存在: {video_file}")
    
    # 显示批量处理结果
    print("\n批量处理结果:")
    for result in results:
        if result['success']:
            print(f"  {result['video_file']}: {result['processing_time']:.2f}秒, "
                  f"{result['frames_processed']}帧, {result['faces_recognized']}个识别")
        else:
            print(f"  {result['video_file']}: 失败")


def main():
    """主函数"""
    print("视频人物识别使用示例")
    print("=" * 50)
    
    try:
        # 运行各种示例
        example_basic_recognition()
        example_realtime_recognition()
        example_performance_optimization()
        example_custom_callback()
        example_batch_processing()
        
        print("\n所有示例运行完成！")
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
