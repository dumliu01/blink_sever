"""
实时视频人物识别
支持多线程处理，提高性能
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Callable
import json
import os
from datetime import datetime
from video_face_recognition import VideoFaceRecognizer


class RealTimeVideoRecognizer:
    """实时视频识别器"""
    
    def __init__(self, model_name: str = 'buffalo_l', 
                 similarity_threshold: float = 0.6,
                 max_queue_size: int = 5,
                 processing_threads: int = 2):
        """
        初始化实时视频识别器
        
        Args:
            model_name: InsightFace模型名称
            similarity_threshold: 人脸识别相似度阈值
            max_queue_size: 最大队列大小
            processing_threads: 处理线程数
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_queue_size = max_queue_size
        self.processing_threads = processing_threads
        
        # 初始化人脸识别器
        self.face_recognizer = VideoFaceRecognizer(
            model_name=model_name,
            similarity_threshold=similarity_threshold
        )
        
        # 线程控制
        self.running = False
        self.capture_thread = None
        self.processing_threads_list = []
        
        # 队列
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        
        # 回调函数
        self.on_face_detected = None
        self.on_face_recognized = None
        self.on_frame_processed = None
        
        # 统计信息
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'faces_recognized': 0,
            'processing_fps': 0,
            'queue_size': 0,
            'start_time': None
        }
        
        print(f"✓ 实时视频识别器初始化完成")
        print(f"  - 模型: {model_name}")
        print(f"  - 识别阈值: {similarity_threshold}")
        print(f"  - 最大队列大小: {max_queue_size}")
        print(f"  - 处理线程数: {processing_threads}")
    
    def set_callbacks(self, on_face_detected: Optional[Callable] = None,
                     on_face_recognized: Optional[Callable] = None,
                     on_frame_processed: Optional[Callable] = None):
        """
        设置回调函数
        
        Args:
            on_face_detected: 人脸检测回调
            on_face_recognized: 人脸识别回调
            on_frame_processed: 帧处理完成回调
        """
        self.on_face_detected = on_face_detected
        self.on_face_recognized = on_face_recognized
        self.on_frame_processed = on_frame_processed
    
    def _capture_frames(self, source):
        """捕获帧的线程函数"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"无法打开视频源: {source}")
            return
        
        # 设置摄像头属性
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 添加到队列
            try:
                self.frame_queue.put((frame_count, frame), timeout=0.1)
            except queue.Full:
                # 队列满时丢弃最旧的帧
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put((frame_count, frame), timeout=0.1)
                except queue.Empty:
                    pass
            
            # 控制帧率
            time.sleep(0.033)  # 约30fps
        
        cap.release()
        print("帧捕获线程结束")
    
    def _process_frames(self, thread_id):
        """处理帧的线程函数"""
        while self.running:
            try:
                # 获取帧
                frame_count, frame = self.frame_queue.get(timeout=1.0)
                
                # 处理帧
                start_time = time.time()
                processed_frame = self.face_recognizer.process_frame(frame)
                processing_time = time.time() - start_time
                
                # 更新统计信息
                self.stats['frames_processed'] += 1
                self.stats['queue_size'] = self.frame_queue.qsize()
                
                # 计算处理FPS
                if self.stats['start_time'] is None:
                    self.stats['start_time'] = time.time()
                else:
                    elapsed_time = time.time() - self.stats['start_time']
                    if elapsed_time > 0:
                        self.stats['processing_fps'] = self.stats['frames_processed'] / elapsed_time
                
                # 触发回调
                if self.on_frame_processed:
                    self.on_frame_processed(processed_frame, frame_count, processing_time)
                
                # 检查识别结果
                if hasattr(self.face_recognizer, 'tracked_faces'):
                    for track_id, face_info in self.face_recognizer.tracked_faces.items():
                        if face_info.get('recognized', False):
                            if self.on_face_recognized:
                                self.on_face_recognized(face_info, frame_count)
                        else:
                            if self.on_face_detected:
                                self.on_face_detected(face_info, frame_count)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理线程 {thread_id} 错误: {e}")
        
        print(f"处理线程 {thread_id} 结束")
    
    def start(self, source=0):
        """
        开始实时识别
        
        Args:
            source: 视频源（摄像头ID或视频文件路径）
        """
        if self.running:
            print("识别器已在运行")
            return
        
        self.running = True
        self.stats['start_time'] = None
        self.stats['frames_processed'] = 0
        
        # 启动捕获线程
        self.capture_thread = threading.Thread(target=self._capture_frames, args=(source,))
        self.capture_thread.start()
        
        # 启动处理线程
        for i in range(self.processing_threads):
            thread = threading.Thread(target=self._process_frames, args=(i,))
            thread.start()
            self.processing_threads_list.append(thread)
        
        print(f"实时识别已启动 (源: {source})")
    
    def stop(self):
        """停止实时识别"""
        if not self.running:
            print("识别器未运行")
            return
        
        self.running = False
        
        # 等待线程结束
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        for thread in self.processing_threads_list:
            thread.join(timeout=2)
        
        self.processing_threads_list.clear()
        
        print("实时识别已停止")
        print(f"统计信息: {self.get_stats()}")
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'frames_processed': self.stats['frames_processed'],
            'faces_detected': self.face_recognizer.stats['total_faces_detected'],
            'faces_recognized': self.face_recognizer.stats['total_faces_recognized'],
            'processing_fps': self.stats['processing_fps'],
            'queue_size': self.stats['queue_size'],
            'unique_persons': list(self.face_recognizer.stats['unique_persons']),
            'recognition_rate': (self.face_recognizer.stats['total_faces_recognized'] / 
                               max(self.face_recognizer.stats['total_faces_detected'], 1) * 100)
        }
    
    def save_recognition_log(self, filepath: str):
        """保存识别日志"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.get_stats(),
            'recognized_persons': list(self.face_recognizer.stats['unique_persons']),
            'settings': {
                'model_name': self.model_name,
                'similarity_threshold': self.similarity_threshold,
                'max_queue_size': self.max_queue_size,
                'processing_threads': self.processing_threads
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"识别日志已保存到: {filepath}")


class VideoRecognitionGUI:
    """视频识别GUI界面"""
    
    def __init__(self, recognizer: RealTimeVideoRecognizer):
        self.recognizer = recognizer
        self.current_frame = None
        self.display_frame = None
        self.window_name = "实时人脸识别"
        
        # 设置回调
        self.recognizer.set_callbacks(
            on_frame_processed=self.on_frame_processed,
            on_face_recognized=self.on_face_recognized,
            on_face_detected=self.on_face_detected
        )
        
        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        # 状态信息
        self.status_text = ""
        self.last_update_time = 0
    
    def on_frame_processed(self, frame, frame_count, processing_time):
        """帧处理完成回调"""
        self.current_frame = frame.copy()
        self.last_update_time = time.time()
    
    def on_face_recognized(self, face_info, frame_count):
        """人脸识别回调"""
        person_name = face_info.get('person_name', 'Unknown')
        similarity = face_info.get('similarity', 0.0)
        track_id = face_info.get('track_id', 0)
        
        print(f"识别到: {person_name} (相似度: {similarity:.3f}, 跟踪ID: {track_id})")
    
    def on_face_detected(self, face_info, frame_count):
        """人脸检测回调"""
        track_id = face_info.get('track_id', 0)
        confidence = face_info.get('confidence', 0.0)
        
        print(f"检测到未知人脸 (跟踪ID: {track_id}, 置信度: {confidence:.3f})")
    
    def update_display(self):
        """更新显示"""
        if self.current_frame is None:
            return
        
        # 创建显示帧
        self.display_frame = self.current_frame.copy()
        
        # 添加状态信息
        stats = self.recognizer.get_stats()
        status_lines = [
            f"FPS: {stats['processing_fps']:.1f}",
            f"Frames: {stats['frames_processed']}",
            f"Faces: {stats['faces_detected']}",
            f"Recognized: {stats['faces_recognized']}",
            f"Queue: {stats['queue_size']}",
            f"Rate: {stats['recognition_rate']:.1f}%"
        ]
        
        # 绘制状态信息
        y_offset = 30
        for line in status_lines:
            cv2.putText(self.display_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # 显示帧
        cv2.imshow(self.window_name, self.display_frame)
    
    def run(self, source=0):
        """运行GUI"""
        print("启动GUI界面...")
        print("控制键:")
        print("  'q' - 退出")
        print("  's' - 保存当前帧")
        print("  'l' - 保存识别日志")
        print("  'r' - 重置统计")
        
        # 启动识别器
        self.recognizer.start(source)
        
        try:
            while True:
                # 更新显示
                self.update_display()
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    if self.current_frame is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"captured_frame_{timestamp}.jpg"
                        cv2.imwrite(filename, self.current_frame)
                        print(f"帧已保存: {filename}")
                elif key == ord('l'):
                    # 保存识别日志
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    logfile = f"recognition_log_{timestamp}.json"
                    self.recognizer.save_recognition_log(logfile)
                elif key == ord('r'):
                    # 重置统计
                    self.recognizer.face_recognizer.reset_stats()
                    print("统计信息已重置")
                
                time.sleep(0.01)  # 避免CPU占用过高
        
        except KeyboardInterrupt:
            print("\n用户中断")
        
        finally:
            # 停止识别器
            self.recognizer.stop()
            cv2.destroyAllWindows()
            print("GUI已关闭")


def demo_real_time_recognition():
    """实时识别演示"""
    print("=== 实时视频人物识别演示 ===")
    
    # 创建实时识别器
    recognizer = RealTimeVideoRecognizer(
        similarity_threshold=0.6,
        max_queue_size=5,
        processing_threads=2
    )
    
    # 注册测试人员
    test_images = {
        "person1": "../test_images/person1_1.jpg",
        "person2": "../test_images/person2_1.jpg",
        "person2_2": "../test_images/person2_2.jpg",
        "person2_3": "../test_images/person2_3.jpg"
    }
    
    print("\n1. 注册测试人员...")
    for person_id, img_path in test_images.items():
        if os.path.exists(img_path):
            person_name = f"人员{person_id}"
            recognizer.face_recognizer.face_recognizer.register_person(person_id, person_name, img_path)
    
    # 显示注册的人员
    print("\n2. 已注册人员:")
    persons = recognizer.face_recognizer.face_recognizer.get_all_persons()
    for person in persons:
        print(f"  {person['person_name']} (ID: {person['person_id']})")
    
    # 选择视频源
    print("\n3. 选择视频源:")
    print("  1. 摄像头 (ID: 0)")
    print("  2. 摄像头 (ID: 1)")
    print("  3. 视频文件")
    print("  4. 退出")
    
    choice = input("请选择 (1-4): ").strip()
    
    if choice in ["1", "2"]:
        # 摄像头
        camera_id = int(choice) - 1
        print(f"\n启动摄像头 {camera_id}...")
        
        # 创建GUI
        gui = VideoRecognitionGUI(recognizer)
        gui.run(camera_id)
        
    elif choice == "3":
        # 视频文件
        video_path = input("请输入视频文件路径: ").strip()
        if os.path.exists(video_path):
            print(f"\n处理视频文件: {video_path}")
            
            # 创建GUI
            gui = VideoRecognitionGUI(recognizer)
            gui.run(video_path)
        else:
            print("视频文件不存在")
    
    elif choice == "4":
        print("退出程序")
    
    else:
        print("无效选择")


if __name__ == "__main__":
    demo_real_time_recognition()
