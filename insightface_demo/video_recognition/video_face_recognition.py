"""
视频人物识别演示
支持实时视频流和视频文件的人物识别
"""

import cv2
import numpy as np
import insightface
from typing import List, Dict, Any, Tuple, Optional
import time
import json
import sqlite3
from collections import defaultdict, deque
from datetime import datetime
import threading
import queue
import os
import sys

# 添加父目录到路径，以便导入face_recognition模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_recognition import FaceRecognizer

class VideoFaceRecognizer:
    """视频人脸识别器"""
    
    def __init__(self, model_name: str = 'buffalo_l', 
                 similarity_threshold: float = 0.6,
                 tracking_threshold: float = 0.5,
                 max_tracking_distance: float = 100.0):
        """
        初始化视频人脸识别器
        
        Args:
            model_name: InsightFace模型名称
            similarity_threshold: 人脸识别相似度阈值
            tracking_threshold: 人脸跟踪相似度阈值
            max_tracking_distance: 最大跟踪距离（像素）
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.tracking_threshold = tracking_threshold
        self.max_tracking_distance = max_tracking_distance
        
        # 初始化人脸识别器
        self.face_recognizer = FaceRecognizer(model_name)
        
        # 人脸跟踪相关
        self.tracked_faces = {}  # {track_id: face_info}
        self.next_track_id = 1
        self.face_history = defaultdict(lambda: deque(maxlen=10))  # 保存最近10帧的人脸信息
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'unique_persons': set(),
            'recognition_times': []
        }
        
        # 性能监控
        self.performance_stats = {
            'fps': 0,
            'avg_processing_time': 0,
            'last_frame_time': 0,
            'recognition_times': []
        }
        
        print(f"✓ 视频人脸识别器初始化完成")
        print(f"  - 模型: {model_name}")
        print(f"  - 识别阈值: {similarity_threshold}")
        print(f"  - 跟踪阈值: {tracking_threshold}")
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        在帧中检测人脸
        
        Args:
            frame: 视频帧
            
        Returns:
            检测到的人脸信息列表
        """
        try:
            faces = self.face_recognizer.app.get(frame)
            detected_faces = []
            
            for i, face in enumerate(faces):
                face_info = {
                    'face_id': i,
                    'bbox': face.bbox.tolist(),
                    'embedding': face.embedding,
                    'confidence': float(face.det_score),
                    'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                    'age': getattr(face, 'age', None),
                    'gender': getattr(face, 'gender', None)
                }
                detected_faces.append(face_info)
            
            return detected_faces
            
        except Exception as e:
            print(f"人脸检测失败: {e}")
            return []
    
    def calculate_face_distance(self, face1: Dict[str, Any], face2: Dict[str, Any]) -> float:
        """
        计算两个人脸之间的距离（基于位置和特征）
        
        Args:
            face1: 人脸1信息
            face2: 人脸2信息
            
        Returns:
            距离值
        """
        # 位置距离（基于边界框中心）
        bbox1 = face1['bbox']
        bbox2 = face2['bbox']
        
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        position_distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # 特征距离
        embedding1 = face1['embedding']
        embedding2 = face2['embedding']
        feature_distance = 1 - np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        # 综合距离（位置权重0.3，特征权重0.7）
        combined_distance = 0.3 * position_distance + 0.7 * feature_distance * 100
        
        return combined_distance
    
    def track_faces(self, detected_faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        跟踪人脸，为检测到的人脸分配跟踪ID
        
        Args:
            detected_faces: 检测到的人脸列表
            
        Returns:
            带跟踪ID的人脸列表
        """
        tracked_faces = []
        
        for detected_face in detected_faces:
            best_match_id = None
            best_distance = float('inf')
            
            # 寻找最佳匹配的已跟踪人脸
            for track_id, tracked_face in self.tracked_faces.items():
                distance = self.calculate_face_distance(detected_face, tracked_face)
                
                if distance < self.max_tracking_distance and distance < best_distance:
                    best_distance = distance
                    best_match_id = track_id
            
            # 如果找到匹配，更新跟踪信息
            if best_match_id is not None:
                detected_face['track_id'] = best_match_id
                detected_face['tracked'] = True
                self.tracked_faces[best_match_id] = detected_face
            else:
                # 创建新的跟踪ID
                detected_face['track_id'] = self.next_track_id
                detected_face['tracked'] = False
                self.tracked_faces[self.next_track_id] = detected_face
                self.next_track_id += 1
            
            tracked_faces.append(detected_face)
        
        # 清理过期的跟踪信息（超过5帧未更新）
        expired_tracks = []
        for track_id, tracked_face in self.tracked_faces.items():
            if 'last_seen' not in tracked_face or time.time() - tracked_face['last_seen'] > 0.5:  # 0.5秒未更新
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del self.tracked_faces[track_id]
        
        return tracked_faces
    
    def recognize_faces(self, tracked_faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        识别跟踪的人脸
        
        Args:
            tracked_faces: 跟踪的人脸列表
            
        Returns:
            识别结果列表
        """
        recognized_faces = []
        
        for face in tracked_faces:
            track_id = face['track_id']
            
            # 检查是否已经识别过这个人脸
            if track_id in self.face_history and len(self.face_history[track_id]) > 0:
                last_recognition = self.face_history[track_id][-1]
                if 'person_id' in last_recognition and time.time() - last_recognition['timestamp'] < 2.0:  # 2秒内不重复识别
                    face.update(last_recognition)
                    recognized_faces.append(face)
                    continue
            
            # 进行人脸识别
            try:
                # 从数据库获取所有注册人员
                conn = sqlite3.connect(self.face_recognizer.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT person_id, person_name, embedding FROM face_embeddings
                ''')
                
                results = cursor.fetchall()
                conn.close()
                
                if not results:
                    face['person_id'] = None
                    face['person_name'] = 'Unknown'
                    face['similarity'] = 0.0
                    face['recognized'] = False
                else:
                    # 计算与所有注册人员的相似度
                    best_match = None
                    max_similarity = 0.0
                    
                    for person_id, person_name, embedding_blob in results:
                        stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                        similarity = np.dot(face['embedding'], stored_embedding) / (
                            np.linalg.norm(face['embedding']) * np.linalg.norm(stored_embedding)
                        )
                        
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match = {
                                'person_id': person_id,
                                'person_name': person_name,
                                'similarity': similarity
                            }
                    
                    if best_match and max_similarity >= self.similarity_threshold:
                        face['person_id'] = best_match['person_id']
                        face['person_name'] = best_match['person_name']
                        face['similarity'] = max_similarity
                        face['recognized'] = True
                        
                        # 更新统计信息
                        self.stats['total_faces_recognized'] += 1
                        self.stats['unique_persons'].add(best_match['person_name'])
                    else:
                        face['person_id'] = None
                        face['person_name'] = 'Unknown'
                        face['similarity'] = max_similarity
                        face['recognized'] = False
                
                # 保存识别历史
                face['timestamp'] = time.time()
                self.face_history[track_id].append(face.copy())
                
                recognized_faces.append(face)
                
            except Exception as e:
                print(f"人脸识别失败: {e}")
                face['person_id'] = None
                face['person_name'] = 'Unknown'
                face['similarity'] = 0.0
                face['recognized'] = False
                recognized_faces.append(face)
        
        return recognized_faces
    
    def draw_face_info(self, frame: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        """
        在帧上绘制人脸信息
        
        Args:
            frame: 视频帧
            faces: 人脸信息列表
            
        Returns:
            绘制后的帧
        """
        for face in faces:
            bbox = face['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # 绘制边界框
            color = (0, 255, 0) if face.get('recognized', False) else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"ID:{face['track_id']}"
            if face.get('recognized', False):
                label += f" {face['person_name']} ({face['similarity']:.2f})"
            else:
                label += " Unknown"
            
            # 绘制置信度
            if 'confidence' in face:
                label += f" Conf:{face['confidence']:.2f}"
            
            # 绘制标签背景
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 绘制标签文字
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_stats(self, frame: np.ndarray) -> np.ndarray:
        """
        在帧上绘制统计信息
        
        Args:
            frame: 视频帧
            
        Returns:
            绘制后的帧
        """
        # 统计信息
        stats_text = [
            f"FPS: {self.performance_stats['fps']:.1f}",
            f"Frames: {self.stats['total_frames']}",
            f"Faces Detected: {self.stats['total_faces_detected']}",
            f"Faces Recognized: {self.stats['total_faces_recognized']}",
            f"Unique Persons: {len(self.stats['unique_persons'])}"
        ]
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制文字
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (20, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧视频
        
        Args:
            frame: 输入帧
            
        Returns:
            处理后的帧
        """
        start_time = time.time()
        
        # 更新统计信息
        self.stats['total_frames'] += 1
        
        # 检测人脸
        detected_faces = self.detect_faces_in_frame(frame)
        self.stats['total_faces_detected'] += len(detected_faces)
        
        # 跟踪人脸
        tracked_faces = self.track_faces(detected_faces)
        
        # 识别人脸
        recognized_faces = self.recognize_faces(tracked_faces)
        
        # 绘制结果
        frame = self.draw_face_info(frame, recognized_faces)
        frame = self.draw_stats(frame)
        
        # 更新性能统计
        processing_time = time.time() - start_time
        self.performance_stats['recognition_times'].append(processing_time)
        if len(self.performance_stats['recognition_times']) > 30:
            self.performance_stats['recognition_times'].pop(0)
        
        self.performance_stats['avg_processing_time'] = np.mean(self.performance_stats['recognition_times'])
        
        if self.performance_stats['last_frame_time'] > 0:
            fps = 1.0 / (time.time() - self.performance_stats['last_frame_time'])
            self.performance_stats['fps'] = fps
        
        self.performance_stats['last_frame_time'] = time.time()
        
        return frame
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None, 
                          display: bool = True) -> bool:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
            display: 是否显示视频窗口
            
        Returns:
            是否处理成功
        """
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频文件: {video_path}")
                return False
            
            # 获取视频属性
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
            
            # 设置输出视频（如果需要）
            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                processed_frame = self.process_frame(frame)
                
                # 保存输出视频
                if out:
                    out.write(processed_frame)
                
                # 显示视频
                if display:
                    cv2.imshow('Video Face Recognition', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"已处理 {frame_count}/{total_frames} 帧")
            
            # 清理资源
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"视频处理完成: {frame_count} 帧")
            print(f"统计信息: {self.get_stats_summary()}")
            
            return True
            
        except Exception as e:
            print(f"视频处理失败: {e}")
            return False
    
    def process_camera(self, camera_id: int = 0, display: bool = True) -> bool:
        """
        处理摄像头视频流
        
        Args:
            camera_id: 摄像头ID
            display: 是否显示视频窗口
            
        Returns:
            是否处理成功
        """
        try:
            # 打开摄像头
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                print(f"无法打开摄像头: {camera_id}")
                return False
            
            # 设置摄像头属性
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"摄像头 {camera_id} 已启动")
            print("按 'q' 键退出")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                processed_frame = self.process_frame(frame)
                
                # 显示视频
                if display:
                    cv2.imshow('Camera Face Recognition', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # 清理资源
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            print("摄像头处理结束")
            print(f"统计信息: {self.get_stats_summary()}")
            
            return True
            
        except Exception as e:
            print(f"摄像头处理失败: {e}")
            return False
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """获取统计信息摘要"""
        return {
            'total_frames': self.stats['total_frames'],
            'total_faces_detected': self.stats['total_faces_detected'],
            'total_faces_recognized': self.stats['total_faces_recognized'],
            'unique_persons': list(self.stats['unique_persons']),
            'recognition_rate': (self.stats['total_faces_recognized'] / 
                               max(self.stats['total_faces_detected'], 1) * 100),
            'avg_processing_time': self.performance_stats['avg_processing_time'],
            'fps': self.performance_stats['fps']
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_frames': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'unique_persons': set(),
            'recognition_times': []
        }
        self.performance_stats = {
            'fps': 0,
            'avg_processing_time': 0,
            'last_frame_time': 0,
            'recognition_times': []
        }
        self.tracked_faces.clear()
        self.face_history.clear()
        self.next_track_id = 1


def demo_video_recognition():
    """视频识别演示"""
    print("=== 视频人物识别演示 ===")
    
    # 创建视频识别器
    recognizer = VideoFaceRecognizer(
        similarity_threshold=0.6,
        tracking_threshold=0.5,
        max_tracking_distance=100.0
    )
    
    # 注册一些测试人员（使用现有的测试图像）
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
            recognizer.face_recognizer.register_person(person_id, person_name, img_path)
    
    # 显示注册的人员
    print("\n2. 已注册人员:")
    persons = recognizer.face_recognizer.get_all_persons()
    for person in persons:
        print(f"  {person['person_name']} (ID: {person['person_id']})")
    
    # 选择处理模式
    print("\n3. 选择处理模式:")
    print("  1. 摄像头实时识别")
    print("  2. 视频文件识别")
    print("  3. 退出")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        # 摄像头实时识别
        print("\n启动摄像头识别...")
        recognizer.process_camera()
        
    elif choice == "2":
        # 视频文件识别
        video_path = input("请输入视频文件路径: ").strip()
        if os.path.exists(video_path):
            output_path = input("请输入输出视频路径（可选，直接回车跳过）: ").strip()
            if not output_path:
                output_path = None
            recognizer.process_video_file(video_path, output_path)
        else:
            print("视频文件不存在")
    
    elif choice == "3":
        print("退出程序")
    
    else:
        print("无效选择")


if __name__ == "__main__":
    demo_video_recognition()
