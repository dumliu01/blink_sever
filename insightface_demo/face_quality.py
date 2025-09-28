"""
InsightFace 人脸质量评估演示
包含人脸质量评估、图像质量分析等功能
"""

import cv2
import numpy as np
import insightface
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib


class FaceQualityAssessor:
    """人脸质量评估器类"""
    
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        初始化人脸质量评估器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.app = None
        self.quality_model = None
        self.scaler = StandardScaler()
        self.load_model()
    
    def load_model(self):
        """加载InsightFace模型"""
        try:
            self.app = insightface.app.FaceAnalysis(name=self.model_name)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print(f"✓ InsightFace模型 {self.model_name} 加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e
    
    def extract_quality_features(self, image_path: str) -> Dict[str, Any]:
        """
        提取人脸质量特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            质量特征字典
        """
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 检测人脸
            faces = self.app.get(img)
            
            if len(faces) == 0:
                return {'error': '未检测到人脸'}
            
            # 使用第一个检测到的人脸
            face = faces[0]
            bbox = face.bbox.astype(int)
            landmarks = face.kps.astype(int)
            
            # 提取人脸区域
            x1, y1, x2, y2 = bbox
            face_region = img[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return {'error': '人脸区域为空'}
            
            # 转换为灰度图像
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # 1. 清晰度特征
            clarity_features = self._extract_clarity_features(gray_face)
            
            # 2. 亮度特征
            brightness_features = self._extract_brightness_features(gray_face)
            
            # 3. 对比度特征
            contrast_features = self._extract_contrast_features(gray_face)
            
            # 4. 对称性特征
            symmetry_features = self._extract_symmetry_features(landmarks, gray_face)
            
            # 5. 角度特征
            angle_features = self._extract_angle_features(landmarks)
            
            # 6. 大小特征
            size_features = self._extract_size_features(bbox, img.shape)
            
            # 7. 遮挡特征
            occlusion_features = self._extract_occlusion_features(landmarks, gray_face)
            
            # 合并所有特征
            features = {
                **clarity_features,
                **brightness_features,
                **contrast_features,
                **symmetry_features,
                **angle_features,
                **size_features,
                **occlusion_features
            }
            
            return features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return {'error': str(e)}
    
    def _extract_clarity_features(self, gray_face: np.ndarray) -> Dict[str, float]:
        """提取清晰度特征"""
        try:
            # 拉普拉斯方差
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # 梯度幅值
            grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_mean = np.mean(gradient_magnitude)
            gradient_std = np.std(gradient_magnitude)
            
            # 边缘密度
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            return {
                'laplacian_var': laplacian_var,
                'gradient_mean': gradient_mean,
                'gradient_std': gradient_std,
                'edge_density': edge_density
            }
        except Exception as e:
            print(f"清晰度特征提取失败: {e}")
            return {'laplacian_var': 0.0, 'gradient_mean': 0.0, 'gradient_std': 0.0, 'edge_density': 0.0}
    
    def _extract_brightness_features(self, gray_face: np.ndarray) -> Dict[str, float]:
        """提取亮度特征"""
        try:
            # 平均亮度
            mean_brightness = np.mean(gray_face)
            
            # 亮度标准差
            brightness_std = np.std(gray_face)
            
            # 亮度分布
            hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # 亮度均匀性（熵）
            brightness_entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # 过曝和欠曝像素比例
            overexposed_ratio = np.sum(gray_face > 240) / gray_face.size
            underexposed_ratio = np.sum(gray_face < 15) / gray_face.size
            
            return {
                'mean_brightness': mean_brightness,
                'brightness_std': brightness_std,
                'brightness_entropy': brightness_entropy,
                'overexposed_ratio': overexposed_ratio,
                'underexposed_ratio': underexposed_ratio
            }
        except Exception as e:
            print(f"亮度特征提取失败: {e}")
            return {'mean_brightness': 0.0, 'brightness_std': 0.0, 'brightness_entropy': 0.0, 
                   'overexposed_ratio': 0.0, 'underexposed_ratio': 0.0}
    
    def _extract_contrast_features(self, gray_face: np.ndarray) -> Dict[str, float]:
        """提取对比度特征"""
        try:
            # 标准差对比度
            contrast_std = np.std(gray_face)
            
            # RMS对比度
            contrast_rms = np.sqrt(np.mean((gray_face - np.mean(gray_face))**2))
            
            # 局部对比度
            kernel = np.ones((3, 3), np.float32) / 9
            local_mean = cv2.filter2D(gray_face.astype(np.float32), -1, kernel)
            local_contrast = np.mean(np.abs(gray_face.astype(np.float32) - local_mean))
            
            # 对比度比率
            min_val = np.min(gray_face)
            max_val = np.max(gray_face)
            contrast_ratio = (max_val - min_val) / (max_val + min_val + 1e-10)
            
            return {
                'contrast_std': contrast_std,
                'contrast_rms': contrast_rms,
                'local_contrast': local_contrast,
                'contrast_ratio': contrast_ratio
            }
        except Exception as e:
            print(f"对比度特征提取失败: {e}")
            return {'contrast_std': 0.0, 'contrast_rms': 0.0, 'local_contrast': 0.0, 'contrast_ratio': 0.0}
    
    def _extract_symmetry_features(self, landmarks: np.ndarray, gray_face: np.ndarray) -> Dict[str, float]:
        """提取对称性特征"""
        try:
            # 计算人脸中心线
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            
            # 眼睛中心
            eye_center = (left_eye + right_eye) / 2
            # 嘴巴中心
            mouth_center = (left_mouth + right_mouth) / 2
            # 人脸中心
            face_center = (eye_center + mouth_center) / 2
            
            # 计算对称性
            h, w = gray_face.shape
            center_x = int(face_center[0])
            
            if center_x < w // 2:
                # 人脸偏左，使用右半部分镜像
                left_half = gray_face[:, :center_x]
                right_half = gray_face[:, center_x:center_x*2]
                if right_half.shape[1] == left_half.shape[1]:
                    right_half_flipped = np.fliplr(right_half)
                    symmetry_score = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half_flipped.astype(float))) / 255.0
                else:
                    symmetry_score = 0.5
            else:
                # 人脸偏右，使用左半部分镜像
                right_half = gray_face[:, center_x:]
                left_half = gray_face[:, center_x*2-w:center_x]
                if left_half.shape[1] == right_half.shape[1]:
                    left_half_flipped = np.fliplr(left_half)
                    symmetry_score = 1.0 - np.mean(np.abs(right_half.astype(float) - left_half_flipped.astype(float))) / 255.0
                else:
                    symmetry_score = 0.5
            
            # 眼睛对称性
            eye_symmetry = 1.0 - abs(left_eye[1] - right_eye[1]) / max(left_eye[1], right_eye[1])
            
            # 嘴巴对称性
            mouth_symmetry = 1.0 - abs(left_mouth[1] - right_mouth[1]) / max(left_mouth[1], right_mouth[1])
            
            return {
                'face_symmetry': symmetry_score,
                'eye_symmetry': eye_symmetry,
                'mouth_symmetry': mouth_symmetry
            }
        except Exception as e:
            print(f"对称性特征提取失败: {e}")
            return {'face_symmetry': 0.5, 'eye_symmetry': 0.5, 'mouth_symmetry': 0.5}
    
    def _extract_angle_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """提取角度特征"""
        try:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            
            # 水平角度
            horizontal_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
            horizontal_angle_deg = np.degrees(horizontal_angle)
            
            # 垂直角度
            eye_center = (left_eye + right_eye) / 2
            vertical_angle = np.arctan2(nose[1] - eye_center[1], nose[0] - eye_center[0])
            vertical_angle_deg = np.degrees(vertical_angle)
            
            # 角度质量评分（越接近0度越好）
            angle_quality = 1.0 - (abs(horizontal_angle_deg) + abs(vertical_angle_deg)) / 180.0
            
            return {
                'horizontal_angle': horizontal_angle_deg,
                'vertical_angle': vertical_angle_deg,
                'angle_quality': angle_quality
            }
        except Exception as e:
            print(f"角度特征提取失败: {e}")
            return {'horizontal_angle': 0.0, 'vertical_angle': 0.0, 'angle_quality': 0.5}
    
    def _extract_size_features(self, bbox: np.ndarray, image_shape: Tuple[int, int, int]) -> Dict[str, float]:
        """提取大小特征"""
        try:
            x1, y1, x2, y2 = bbox
            face_width = x2 - x1
            face_height = y2 - y1
            face_area = face_width * face_height
            
            image_height, image_width = image_shape[:2]
            image_area = image_height * image_width
            
            # 人脸在图像中的比例
            size_ratio = face_area / image_area
            
            # 人脸宽高比
            aspect_ratio = face_width / face_height if face_height > 0 else 1.0
            
            # 人脸大小评分
            if size_ratio > 0.01:
                size_score = 1.0
            elif size_ratio > 0.005:
                size_score = 0.8
            elif size_ratio > 0.002:
                size_score = 0.6
            else:
                size_score = 0.3
            
            return {
                'face_width': face_width,
                'face_height': face_height,
                'face_area': face_area,
                'size_ratio': size_ratio,
                'aspect_ratio': aspect_ratio,
                'size_score': size_score
            }
        except Exception as e:
            print(f"大小特征提取失败: {e}")
            return {'face_width': 0, 'face_height': 0, 'face_area': 0, 
                   'size_ratio': 0.0, 'aspect_ratio': 1.0, 'size_score': 0.0}
    
    def _extract_occlusion_features(self, landmarks: np.ndarray, gray_face: np.ndarray) -> Dict[str, float]:
        """提取遮挡特征"""
        try:
            # 基于关键点检测遮挡
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            
            # 眼睛区域遮挡检测
            eye_region_size = 20
            left_eye_region = gray_face[
                max(0, left_eye[1] - eye_region_size):min(gray_face.shape[0], left_eye[1] + eye_region_size),
                max(0, left_eye[0] - eye_region_size):min(gray_face.shape[1], left_eye[0] + eye_region_size)
            ]
            right_eye_region = gray_face[
                max(0, right_eye[1] - eye_region_size):min(gray_face.shape[0], right_eye[1] + eye_region_size),
                max(0, right_eye[0] - eye_region_size):min(gray_face.shape[1], right_eye[0] + eye_region_size)
            ]
            
            # 嘴巴区域遮挡检测
            mouth_center = (left_mouth + right_mouth) / 2
            mouth_region_size = 30
            mouth_region = gray_face[
                max(0, int(mouth_center[1]) - mouth_region_size):min(gray_face.shape[0], int(mouth_center[1]) + mouth_region_size),
                max(0, int(mouth_center[0]) - mouth_region_size):min(gray_face.shape[1], int(mouth_center[0]) + mouth_region_size)
            ]
            
            # 计算遮挡分数（基于区域内的边缘密度）
            def calculate_occlusion_score(region):
                if region.size == 0:
                    return 0.0
                edges = cv2.Canny(region, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                return min(1.0, edge_density * 10)  # 归一化到0-1
            
            left_eye_occlusion = calculate_occlusion_score(left_eye_region)
            right_eye_occlusion = calculate_occlusion_score(right_eye_region)
            mouth_occlusion = calculate_occlusion_score(mouth_region)
            
            # 总体遮挡分数
            total_occlusion = (left_eye_occlusion + right_eye_occlusion + mouth_occlusion) / 3.0
            
            return {
                'left_eye_occlusion': left_eye_occlusion,
                'right_eye_occlusion': right_eye_occlusion,
                'mouth_occlusion': mouth_occlusion,
                'total_occlusion': total_occlusion
            }
        except Exception as e:
            print(f"遮挡特征提取失败: {e}")
            return {'left_eye_occlusion': 0.0, 'right_eye_occlusion': 0.0, 
                   'mouth_occlusion': 0.0, 'total_occlusion': 0.0}
    
    def assess_quality(self, image_path: str) -> Dict[str, Any]:
        """
        评估人脸质量
        
        Args:
            image_path: 图像路径
            
        Returns:
            质量评估结果
        """
        try:
            # 提取质量特征
            features = self.extract_quality_features(image_path)
            
            if 'error' in features:
                return {'quality_score': 0.0, 'quality_level': '错误', 'error': features['error']}
            
            # 计算综合质量评分
            quality_score = self._calculate_quality_score(features)
            
            # 确定质量等级
            quality_level = self._determine_quality_level(quality_score)
            
            # 识别质量问题
            issues = self._identify_quality_issues(features)
            
            return {
                'quality_score': quality_score,
                'quality_level': quality_level,
                'issues': issues,
                'features': features
            }
            
        except Exception as e:
            print(f"质量评估失败: {e}")
            return {'quality_score': 0.0, 'quality_level': '错误', 'error': str(e)}
    
    def _calculate_quality_score(self, features: Dict[str, Any]) -> float:
        """计算综合质量评分"""
        try:
            score = 0.0
            weights = {
                'clarity': 0.25,
                'brightness': 0.20,
                'contrast': 0.15,
                'symmetry': 0.15,
                'angle': 0.10,
                'size': 0.10,
                'occlusion': 0.05
            }
            
            # 清晰度评分
            clarity_score = min(1.0, features.get('laplacian_var', 0) / 100.0)
            score += clarity_score * weights['clarity']
            
            # 亮度评分
            brightness = features.get('mean_brightness', 0)
            if 80 <= brightness <= 200:
                brightness_score = 1.0
            elif 50 <= brightness < 80 or 200 < brightness <= 250:
                brightness_score = 0.7
            else:
                brightness_score = 0.3
            score += brightness_score * weights['brightness']
            
            # 对比度评分
            contrast_score = min(1.0, features.get('contrast_std', 0) / 50.0)
            score += contrast_score * weights['contrast']
            
            # 对称性评分
            symmetry_score = features.get('face_symmetry', 0.5)
            score += symmetry_score * weights['symmetry']
            
            # 角度评分
            angle_score = features.get('angle_quality', 0.5)
            score += angle_score * weights['angle']
            
            # 大小评分
            size_score = features.get('size_score', 0.0)
            score += size_score * weights['size']
            
            # 遮挡评分（遮挡越少越好）
            occlusion_score = 1.0 - features.get('total_occlusion', 0.0)
            score += occlusion_score * weights['occlusion']
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            print(f"质量评分计算失败: {e}")
            return 0.0
    
    def _determine_quality_level(self, score: float) -> str:
        """确定质量等级"""
        if score >= 0.8:
            return '优秀'
        elif score >= 0.6:
            return '良好'
        elif score >= 0.4:
            return '一般'
        else:
            return '差'
    
    def _identify_quality_issues(self, features: Dict[str, Any]) -> List[str]:
        """识别质量问题"""
        issues = []
        
        # 清晰度问题
        if features.get('laplacian_var', 0) < 50:
            issues.append('图像模糊')
        
        # 亮度问题
        brightness = features.get('mean_brightness', 0)
        if brightness < 50:
            issues.append('图像过暗')
        elif brightness > 200:
            issues.append('图像过亮')
        
        # 对比度问题
        if features.get('contrast_std', 0) < 20:
            issues.append('对比度不足')
        
        # 角度问题
        if features.get('angle_quality', 0.5) < 0.7:
            issues.append('人脸角度不佳')
        
        # 大小问题
        if features.get('size_score', 0) < 0.6:
            issues.append('人脸过小')
        
        # 遮挡问题
        if features.get('total_occlusion', 0) > 0.3:
            issues.append('人脸被遮挡')
        
        return issues
    
    def visualize_quality_assessment(self, image_path: str, save_path: str = None) -> np.ndarray:
        """
        可视化质量评估结果
        
        Args:
            image_path: 图像路径
            save_path: 保存路径（可选）
            
        Returns:
            可视化结果图像
        """
        try:
            img = cv2.imread(image_path)
            assessment = self.assess_quality(image_path)
            
            # 检测人脸
            faces = self.app.get(img)
            if len(faces) == 0:
                return img
            
            face = faces[0]
            bbox = face.bbox.astype(int)
            
            # 绘制边界框
            color = (0, 255, 0) if assessment['quality_score'] > 0.6 else (0, 0, 255)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 绘制质量信息
            info_text = [
                f"质量评分: {assessment['quality_score']:.3f}",
                f"质量等级: {assessment['quality_level']}",
                f"问题: {', '.join(assessment['issues']) if assessment['issues'] else '无'}"
            ]
            
            # 在图像上绘制文本
            y_offset = bbox[1] - 10
            for i, text in enumerate(info_text):
                cv2.putText(img, text, (bbox[0], y_offset - i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 保存结果
            if save_path:
                cv2.imwrite(save_path, img)
                print(f"质量评估结果已保存到: {save_path}")
            
            return img
            
        except Exception as e:
            print(f"可视化失败: {e}")
            return None


def demo_face_quality():
    """人脸质量评估演示"""
    print("=== InsightFace 人脸质量评估演示 ===")
    
    # 创建质量评估器
    assessor = FaceQualityAssessor()
    
    # 测试图像路径
    test_images = [
        "test_images/person1_1.jpg",
        "test_images/person2_1.jpg",
        "test_images/person2_2.jpg",
        "test_images/person2_3.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n评估图像: {img_path}")
            
            # 评估人脸质量
            assessment = assessor.assess_quality(img_path)
            
            print(f"质量评分: {assessment['quality_score']:.3f}")
            print(f"质量等级: {assessment['quality_level']}")
            
            if assessment['issues']:
                print(f"质量问题: {', '.join(assessment['issues'])}")
            else:
                print("无质量问题")
            
            # 显示详细特征
            features = assessment.get('features', {})
            if features:
                print("\n详细特征:")
                print(f"  清晰度 (拉普拉斯方差): {features.get('laplacian_var', 0):.2f}")
                print(f"  平均亮度: {features.get('mean_brightness', 0):.2f}")
                print(f"  对比度标准差: {features.get('contrast_std', 0):.2f}")
                print(f"  人脸对称性: {features.get('face_symmetry', 0):.3f}")
                print(f"  角度质量: {features.get('angle_quality', 0):.3f}")
                print(f"  大小评分: {features.get('size_score', 0):.3f}")
                print(f"  遮挡程度: {features.get('total_occlusion', 0):.3f}")
            
            # 可视化结果
            output_path = f"output/quality_{os.path.basename(img_path)}"
            os.makedirs("output", exist_ok=True)
            assessor.visualize_quality_assessment(img_path, output_path)
        else:
            print(f"图像不存在: {img_path}")


if __name__ == "__main__":
    demo_face_quality()
