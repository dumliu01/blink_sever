"""
InsightFace 人脸检测演示
包含人脸检测、关键点检测、人脸对齐等功能
"""

import cv2
import numpy as np
import insightface
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import os


class FaceDetector:
    """人脸检测器类"""
    
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        初始化人脸检测器
        
        Args:
            model_name: 模型名称，可选 'buffalo_l', 'buffalo_m', 'buffalo_s'
        """
        self.model_name = model_name
        self.app = None
        self.load_model()
    
    def load_model(self):
        """加载InsightFace模型"""
        try:
            self.app = insightface.app.FaceAnalysis(name=self.model_name)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print(f"✓ InsightFace模型 {self.model_name} 加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("提示：如果是网络问题，请检查网络连接或使用代理")
            raise e
    
    def detect_faces(self, image_path: str) -> List[Dict[str, Any]]:
        """
        检测图像中的人脸
        
        Args:
            image_path: 图像路径
            
        Returns:
            人脸检测结果列表
        """
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 检测人脸
            faces = self.app.get(img)
            
            results = []
            for i, face in enumerate(faces):
                # 获取边界框
                bbox = face.bbox.astype(int)
                
                # 获取关键点
                landmarks = face.kps.astype(int)
                
                # 获取置信度
                confidence = float(face.det_score)
                
                # 获取人脸角度
                angle = self._calculate_face_angle(landmarks)
                
                results.append({
                    'face_id': i,
                    'bbox': bbox.tolist(),
                    'landmarks': landmarks.tolist(),
                    'confidence': confidence,
                    'angle': angle,
                    'area': self._calculate_face_area(bbox)
                })
            
            return results
            
        except Exception as e:
            print(f"人脸检测失败: {e}")
            raise e
    
    def _calculate_face_angle(self, landmarks: np.ndarray) -> float:
        """计算人脸角度"""
        # 使用眼睛关键点计算角度
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # 计算角度
        angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        return np.degrees(angle)
    
    def _calculate_face_area(self, bbox: np.ndarray) -> float:
        """计算人脸面积"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def align_face(self, image_path: str, face_index: int = 0) -> np.ndarray:
        """
        人脸对齐
        
        Args:
            image_path: 图像路径
            face_index: 人脸索引
            
        Returns:
            对齐后的人脸图像
        """
        try:
            img = cv2.imread(image_path)
            faces = self.app.get(img)
            
            if face_index >= len(faces):
                raise ValueError(f"人脸索引 {face_index} 超出范围")
            
            face = faces[face_index]
            
            # 使用InsightFace的人脸对齐功能
            aligned_face = insightface.utils.face_align.norm_crop(img, landmark=face.kps, image_size=112)
            
            return aligned_face
            
        except Exception as e:
            print(f"人脸对齐失败: {e}")
            raise e
    
    def extract_face_region(self, image_path: str, face_index: int = 0, padding: float = 0.2) -> np.ndarray:
        """
        提取人脸区域
        
        Args:
            image_path: 图像路径
            face_index: 人脸索引
            padding: 边界框扩展比例
            
        Returns:
            人脸区域图像
        """
        try:
            img = cv2.imread(image_path)
            faces = self.app.get(img)
            
            if face_index >= len(faces):
                raise ValueError(f"人脸索引 {face_index} 超出范围")
            
            face = faces[face_index]
            bbox = face.bbox.astype(int)
            
            # 计算扩展后的边界框
            x1, y1, x2, y2 = bbox
            h, w = img.shape[:2]
            
            # 计算扩展量
            expand_w = int((x2 - x1) * padding)
            expand_h = int((y2 - y1) * padding)
            
            # 扩展边界框
            x1 = max(0, x1 - expand_w)
            y1 = max(0, y1 - expand_h)
            x2 = min(w, x2 + expand_w)
            y2 = min(h, y2 + expand_h)
            
            # 提取人脸区域
            face_region = img[y1:y2, x1:x2]
            
            return face_region
            
        except Exception as e:
            print(f"提取人脸区域失败: {e}")
            raise e
    
    def visualize_detection(self, image_path: str, save_path: str = None) -> np.ndarray:
        """
        可视化人脸检测结果
        
        Args:
            image_path: 图像路径
            save_path: 保存路径（可选）
            
        Returns:
            可视化结果图像
        """
        try:
            img = cv2.imread(image_path)
            faces = self.detect_faces(image_path)
            
            # 绘制检测结果
            for face in faces:
                bbox = face['bbox']
                landmarks = face['landmarks']
                confidence = face['confidence']
                
                # 绘制边界框
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # 绘制置信度
                cv2.putText(img, f"{confidence:.3f}", (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 绘制关键点
                for point in landmarks:
                    cv2.circle(img, tuple(point), 3, (255, 0, 0), -1)
            
            # 保存结果
            if save_path:
                cv2.imwrite(save_path, img)
                print(f"检测结果已保存到: {save_path}")
            
            return img
            
        except Exception as e:
            print(f"可视化失败: {e}")
            raise e


def demo_face_detection():
    """人脸检测演示"""
    print("=== InsightFace 人脸检测演示 ===")
    
    # 创建检测器
    detector = FaceDetector()
    
    # 测试图像路径（需要用户提供）
    test_images = [
        "test_images/person1_1.jpg",
        "test_images/person2_1.jpg",
        "test_images/person2_2.jpg",
        "test_images/person2_3.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n处理图像: {img_path}")
            
            # 检测人脸
            faces = detector.detect_faces(img_path)
            print(f"检测到 {len(faces)} 个人脸")
            
            # 显示详细信息
            for i, face in enumerate(faces):
                print(f"  人脸 {i+1}:")
                print(f"    置信度: {face['confidence']:.3f}")
                print(f"    边界框: {face['bbox']}")
                print(f"    角度: {face['angle']:.1f}°")
                print(f"    面积: {face['area']}")
            
            # 可视化结果
            output_path = f"output/detection_{os.path.basename(img_path)}"
            os.makedirs("output", exist_ok=True)
            detector.visualize_detection(img_path, output_path)
        else:
            print(f"图像不存在: {img_path}")


if __name__ == "__main__":
    demo_face_detection()
