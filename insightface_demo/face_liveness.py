"""
InsightFace 人脸活体检测演示
包含活体检测、反欺骗检测等功能
"""

import cv2
import numpy as np
import insightface
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


class FaceLivenessDetector:
    """人脸活体检测器类"""
    
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        初始化人脸活体检测器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.app = None
        self.liveness_model = None
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
    
    def extract_liveness_features(self, image_path: str) -> Dict[str, Any]:
        """
        提取活体检测特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            活体检测特征字典
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
            
            # 1. 纹理特征
            texture_features = self._extract_texture_features(gray_face)
            
            # 2. 频率域特征
            frequency_features = self._extract_frequency_features(gray_face)
            
            # 3. 边缘特征
            edge_features = self._extract_edge_features(gray_face)
            
            # 4. 颜色特征
            color_features = self._extract_color_features(face_region)
            
            # 5. 深度特征（基于人脸几何）
            depth_features = self._extract_depth_features(landmarks, gray_face)
            
            # 6. 运动特征（需要多帧图像，这里使用单帧近似）
            motion_features = self._extract_motion_features(gray_face)
            
            # 7. 反射特征
            reflection_features = self._extract_reflection_features(face_region)
            
            # 合并所有特征
            features = {
                **texture_features,
                **frequency_features,
                **edge_features,
                **color_features,
                **depth_features,
                **motion_features,
                **reflection_features
            }
            
            return features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return {'error': str(e)}
    
    def _extract_texture_features(self, gray_face: np.ndarray) -> Dict[str, float]:
        """提取纹理特征"""
        try:
            # 局部二值模式 (LBP) 特征
            lbp_features = self._calculate_lbp_features(gray_face)
            
            # 灰度共生矩阵 (GLCM) 特征
            glcm_features = self._calculate_glcm_features(gray_face)
            
            # 小波变换特征
            wavelet_features = self._calculate_wavelet_features(gray_face)
            
            return {
                **lbp_features,
                **glcm_features,
                **wavelet_features
            }
        except Exception as e:
            print(f"纹理特征提取失败: {e}")
            return {}
    
    def _calculate_lbp_features(self, gray_face: np.ndarray) -> Dict[str, float]:
        """计算LBP特征"""
        try:
            # 简化的LBP实现
            h, w = gray_face.shape
            lbp_image = np.zeros_like(gray_face)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray_face[i, j]
                    lbp_code = 0
                    for k, (di, dj) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                                                (1, 1), (1, 0), (1, -1), (0, -1)]):
                        if gray_face[i+di, j+dj] >= center:
                            lbp_code |= (1 << k)
                    lbp_image[i, j] = lbp_code
            
            # 计算LBP直方图
            hist, _ = np.histogram(lbp_image, bins=256, range=(0, 256))
            hist = hist.astype(float) / hist.sum()
            
            # 计算LBP统计特征
            lbp_mean = np.mean(lbp_image)
            lbp_std = np.std(lbp_image)
            lbp_entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            return {
                'lbp_mean': lbp_mean,
                'lbp_std': lbp_std,
                'lbp_entropy': lbp_entropy
            }
        except Exception as e:
            print(f"LBP特征计算失败: {e}")
            return {'lbp_mean': 0.0, 'lbp_std': 0.0, 'lbp_entropy': 0.0}
    
    def _calculate_glcm_features(self, gray_face: np.ndarray) -> Dict[str, float]:
        """计算GLCM特征"""
        try:
            # 简化的GLCM实现
            # 将图像量化为8个灰度级
            quantized = (gray_face // 32).astype(np.uint8)
            
            # 计算水平方向的GLCM
            glcm = np.zeros((8, 8), dtype=np.float32)
            h, w = quantized.shape
            
            for i in range(h):
                for j in range(w-1):
                    glcm[quantized[i, j], quantized[i, j+1]] += 1
            
            # 归一化
            glcm = glcm / glcm.sum()
            
            # 计算GLCM特征
            contrast = 0
            homogeneity = 0
            energy = 0
            correlation = 0
            
            for i in range(8):
                for j in range(8):
                    contrast += glcm[i, j] * (i - j) ** 2
                    homogeneity += glcm[i, j] / (1 + abs(i - j))
                    energy += glcm[i, j] ** 2
                    correlation += glcm[i, j] * i * j
            
            return {
                'glcm_contrast': contrast,
                'glcm_homogeneity': homogeneity,
                'glcm_energy': energy,
                'glcm_correlation': correlation
            }
        except Exception as e:
            print(f"GLCM特征计算失败: {e}")
            return {'glcm_contrast': 0.0, 'glcm_homogeneity': 0.0, 
                   'glcm_energy': 0.0, 'glcm_correlation': 0.0}
    
    def _calculate_wavelet_features(self, gray_face: np.ndarray) -> Dict[str, float]:
        """计算小波变换特征"""
        try:
            # 简化的Haar小波变换
            def haar_transform(img):
                h, w = img.shape
                # 水平变换
                img_h = np.zeros_like(img)
                for i in range(h):
                    for j in range(0, w-1, 2):
                        img_h[i, j//2] = (img[i, j] + img[i, j+1]) / 2
                        img_h[i, j//2 + w//2] = (img[i, j] - img[i, j+1]) / 2
                
                # 垂直变换
                img_v = np.zeros_like(img)
                for i in range(0, h-1, 2):
                    for j in range(w):
                        img_v[i//2, j] = (img_h[i, j] + img_h[i+1, j]) / 2
                        img_v[i//2 + h//2, j] = (img_h[i, j] - img_h[i+1, j]) / 2
                
                return img_v
            
            # 执行小波变换
            wavelet_img = haar_transform(gray_face)
            
            # 计算小波系数统计特征
            h, w = wavelet_img.shape
            h_half, w_half = h // 2, w // 2
            
            # 低频分量 (LL)
            ll = wavelet_img[:h_half, :w_half]
            # 高频分量 (LH, HL, HH)
            lh = wavelet_img[:h_half, w_half:]
            hl = wavelet_img[h_half:, :w_half]
            hh = wavelet_img[h_half:, w_half:]
            
            # 计算各分量的能量
            ll_energy = np.sum(ll ** 2)
            lh_energy = np.sum(lh ** 2)
            hl_energy = np.sum(hl ** 2)
            hh_energy = np.sum(hh ** 2)
            
            total_energy = ll_energy + lh_energy + hl_energy + hh_energy
            
            return {
                'wavelet_ll_energy': ll_energy / total_energy if total_energy > 0 else 0,
                'wavelet_lh_energy': lh_energy / total_energy if total_energy > 0 else 0,
                'wavelet_hl_energy': hl_energy / total_energy if total_energy > 0 else 0,
                'wavelet_hh_energy': hh_energy / total_energy if total_energy > 0 else 0
            }
        except Exception as e:
            print(f"小波特征计算失败: {e}")
            return {'wavelet_ll_energy': 0.0, 'wavelet_lh_energy': 0.0, 
                   'wavelet_hl_energy': 0.0, 'wavelet_hh_energy': 0.0}
    
    def _extract_frequency_features(self, gray_face: np.ndarray) -> Dict[str, float]:
        """提取频率域特征"""
        try:
            # 傅里叶变换
            f_transform = np.fft.fft2(gray_face)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # 计算频率域特征
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # 低频能量
            low_freq_region = magnitude_spectrum[center_h-h//4:center_h+h//4, 
                                               center_w-w//4:center_w+w//4]
            low_freq_energy = np.sum(low_freq_region)
            
            # 高频能量
            high_freq_mask = np.ones_like(magnitude_spectrum)
            high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 0
            high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
            
            # 总能量
            total_energy = low_freq_energy + high_freq_energy
            
            # 频率分布特征
            freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
            
            return {
                'low_freq_energy': low_freq_energy,
                'high_freq_energy': high_freq_energy,
                'freq_ratio': freq_ratio
            }
        except Exception as e:
            print(f"频率特征提取失败: {e}")
            return {'low_freq_energy': 0.0, 'high_freq_energy': 0.0, 'freq_ratio': 0.0}
    
    def _extract_edge_features(self, gray_face: np.ndarray) -> Dict[str, float]:
        """提取边缘特征"""
        try:
            # Canny边缘检测
            edges = cv2.Canny(gray_face, 50, 150)
            
            # 边缘密度
            edge_density = np.sum(edges > 0) / edges.size
            
            # 边缘方向分布
            grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_direction = np.arctan2(grad_y, grad_x)
            
            # 边缘方向统计
            direction_std = np.std(gradient_direction[gradient_magnitude > np.percentile(gradient_magnitude, 90)])
            
            # 边缘连续性
            edge_continuity = self._calculate_edge_continuity(edges)
            
            return {
                'edge_density': edge_density,
                'direction_std': direction_std,
                'edge_continuity': edge_continuity
            }
        except Exception as e:
            print(f"边缘特征提取失败: {e}")
            return {'edge_density': 0.0, 'direction_std': 0.0, 'edge_continuity': 0.0}
    
    def _calculate_edge_continuity(self, edges: np.ndarray) -> float:
        """计算边缘连续性"""
        try:
            # 使用形态学操作计算边缘连续性
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            # 连续性 = 原始边缘与形态学操作后边缘的交集比例
            intersection = np.sum((edges > 0) & (eroded > 0))
            total_edges = np.sum(edges > 0)
            
            continuity = intersection / total_edges if total_edges > 0 else 0
            return continuity
        except Exception as e:
            print(f"边缘连续性计算失败: {e}")
            return 0.0
    
    def _extract_color_features(self, face_region: np.ndarray) -> Dict[str, float]:
        """提取颜色特征"""
        try:
            # 转换到不同颜色空间
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            # HSV特征
            h_mean, h_std = np.mean(hsv[:, :, 0]), np.std(hsv[:, :, 0])
            s_mean, s_std = np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1])
            v_mean, v_std = np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])
            
            # LAB特征
            l_mean, l_std = np.mean(lab[:, :, 0]), np.std(lab[:, :, 0])
            a_mean, a_std = np.mean(lab[:, :, 1]), np.std(lab[:, :, 1])
            b_mean, b_std = np.mean(lab[:, :, 2]), np.std(lab[:, :, 2])
            
            # 肤色检测
            skin_ratio = self._detect_skin_ratio(face_region)
            
            return {
                'h_mean': h_mean, 'h_std': h_std,
                's_mean': s_mean, 's_std': s_std,
                'v_mean': v_mean, 'v_std': v_std,
                'l_mean': l_mean, 'l_std': l_std,
                'a_mean': a_mean, 'a_std': a_std,
                'b_mean': b_mean, 'b_std': b_std,
                'skin_ratio': skin_ratio
            }
        except Exception as e:
            print(f"颜色特征提取失败: {e}")
            return {'h_mean': 0.0, 'h_std': 0.0, 's_mean': 0.0, 's_std': 0.0,
                   'v_mean': 0.0, 'v_std': 0.0, 'l_mean': 0.0, 'l_std': 0.0,
                   'a_mean': 0.0, 'a_std': 0.0, 'b_mean': 0.0, 'b_std': 0.0,
                   'skin_ratio': 0.0}
    
    def _detect_skin_ratio(self, face_region: np.ndarray) -> float:
        """检测肤色比例"""
        try:
            # 定义肤色范围 (HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            return skin_ratio
        except Exception as e:
            print(f"肤色检测失败: {e}")
            return 0.0
    
    def _extract_depth_features(self, landmarks: np.ndarray, gray_face: np.ndarray) -> Dict[str, float]:
        """提取深度特征（基于人脸几何）"""
        try:
            # 计算人脸关键点之间的距离和角度
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            
            # 眼睛距离
            eye_distance = np.linalg.norm(right_eye - left_eye)
            
            # 鼻子到眼睛中心的距离
            eye_center = (left_eye + right_eye) / 2
            nose_eye_distance = np.linalg.norm(nose - eye_center)
            
            # 嘴巴宽度
            mouth_width = np.linalg.norm(right_mouth - left_mouth)
            
            # 人脸比例特征
            face_ratio = nose_eye_distance / eye_distance if eye_distance > 0 else 0
            mouth_eye_ratio = mouth_width / eye_distance if eye_distance > 0 else 0
            
            # 基于几何特征的深度估计
            depth_score = self._estimate_depth_from_geometry(landmarks)
            
            return {
                'eye_distance': eye_distance,
                'nose_eye_distance': nose_eye_distance,
                'mouth_width': mouth_width,
                'face_ratio': face_ratio,
                'mouth_eye_ratio': mouth_eye_ratio,
                'depth_score': depth_score
            }
        except Exception as e:
            print(f"深度特征提取失败: {e}")
            return {'eye_distance': 0.0, 'nose_eye_distance': 0.0, 'mouth_width': 0.0,
                   'face_ratio': 0.0, 'mouth_eye_ratio': 0.0, 'depth_score': 0.0}
    
    def _estimate_depth_from_geometry(self, landmarks: np.ndarray) -> float:
        """基于几何特征估计深度"""
        try:
            # 简化的深度估计，基于关键点的几何关系
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            
            # 计算三角形面积（眼睛和鼻子）
            area = 0.5 * abs((left_eye[0] - nose[0]) * (right_eye[1] - nose[1]) - 
                            (right_eye[0] - nose[0]) * (left_eye[1] - nose[1]))
            
            # 计算眼睛距离
            eye_distance = np.linalg.norm(right_eye - left_eye)
            
            # 深度评分（面积与距离的比值）
            depth_score = area / (eye_distance + 1e-10)
            
            return depth_score
        except Exception as e:
            print(f"深度估计失败: {e}")
            return 0.0
    
    def _extract_motion_features(self, gray_face: np.ndarray) -> Dict[str, float]:
        """提取运动特征（单帧近似）"""
        try:
            # 使用光流估计运动
            # 这里使用简化的方法，基于图像梯度
            grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            
            # 运动强度
            motion_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            motion_mean = np.mean(motion_magnitude)
            motion_std = np.std(motion_magnitude)
            
            # 运动方向一致性
            motion_direction = np.arctan2(grad_y, grad_x)
            direction_consistency = 1.0 - np.std(motion_direction) / np.pi
            
            return {
                'motion_mean': motion_mean,
                'motion_std': motion_std,
                'direction_consistency': direction_consistency
            }
        except Exception as e:
            print(f"运动特征提取失败: {e}")
            return {'motion_mean': 0.0, 'motion_std': 0.0, 'direction_consistency': 0.0}
    
    def _extract_reflection_features(self, face_region: np.ndarray) -> Dict[str, float]:
        """提取反射特征"""
        try:
            # 检测镜面反射
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # 高亮区域检测
            bright_threshold = 200
            bright_pixels = np.sum(gray > bright_threshold)
            bright_ratio = bright_pixels / gray.size
            
            # 反射区域检测
            reflection_threshold = 240
            reflection_pixels = np.sum(gray > reflection_threshold)
            reflection_ratio = reflection_pixels / gray.size
            
            # 反射均匀性
            bright_mask = gray > bright_threshold
            if np.sum(bright_mask) > 0:
                bright_region = gray[bright_mask]
                reflection_uniformity = 1.0 - np.std(bright_region) / np.mean(bright_region)
            else:
                reflection_uniformity = 0.0
            
            return {
                'bright_ratio': bright_ratio,
                'reflection_ratio': reflection_ratio,
                'reflection_uniformity': reflection_uniformity
            }
        except Exception as e:
            print(f"反射特征提取失败: {e}")
            return {'bright_ratio': 0.0, 'reflection_ratio': 0.0, 'reflection_uniformity': 0.0}
    
    def detect_liveness(self, image_path: str) -> Dict[str, Any]:
        """
        检测活体
        
        Args:
            image_path: 图像路径
            
        Returns:
            活体检测结果
        """
        try:
            # 提取活体检测特征
            features = self.extract_liveness_features(image_path)
            
            if 'error' in features:
                return {'is_live': False, 'confidence': 0.0, 'error': features['error']}
            
            # 计算活体评分
            liveness_score = self._calculate_liveness_score(features)
            
            # 确定活体状态
            is_live = liveness_score > 0.5
            confidence = abs(liveness_score - 0.5) * 2  # 转换为0-1的置信度
            
            # 识别欺骗类型
            spoof_type = self._identify_spoof_type(features)
            
            return {
                'is_live': is_live,
                'confidence': confidence,
                'liveness_score': liveness_score,
                'spoof_type': spoof_type,
                'features': features
            }
            
        except Exception as e:
            print(f"活体检测失败: {e}")
            return {'is_live': False, 'confidence': 0.0, 'error': str(e)}
    
    def _calculate_liveness_score(self, features: Dict[str, Any]) -> float:
        """计算活体评分"""
        try:
            score = 0.0
            weights = {
                'texture': 0.25,
                'frequency': 0.20,
                'edge': 0.15,
                'color': 0.15,
                'depth': 0.15,
                'motion': 0.05,
                'reflection': 0.05
            }
            
            # 纹理特征评分
            lbp_entropy = features.get('lbp_entropy', 0)
            glcm_contrast = features.get('glcm_contrast', 0)
            texture_score = min(1.0, (lbp_entropy + glcm_contrast) / 10.0)
            score += texture_score * weights['texture']
            
            # 频率特征评分
            freq_ratio = features.get('freq_ratio', 0)
            frequency_score = 1.0 - abs(freq_ratio - 0.5) * 2  # 接近0.5最好
            score += frequency_score * weights['frequency']
            
            # 边缘特征评分
            edge_density = features.get('edge_density', 0)
            edge_continuity = features.get('edge_continuity', 0)
            edge_score = min(1.0, edge_density * edge_continuity * 5)
            score += edge_score * weights['edge']
            
            # 颜色特征评分
            skin_ratio = features.get('skin_ratio', 0)
            color_score = min(1.0, skin_ratio * 2)  # 肤色比例越高越好
            score += color_score * weights['color']
            
            # 深度特征评分
            depth_score = features.get('depth_score', 0)
            score += min(1.0, depth_score * 10) * weights['depth']
            
            # 运动特征评分
            motion_mean = features.get('motion_mean', 0)
            motion_score = min(1.0, motion_mean / 50.0)
            score += motion_score * weights['motion']
            
            # 反射特征评分
            reflection_ratio = features.get('reflection_ratio', 0)
            reflection_score = 1.0 - min(1.0, reflection_ratio * 5)  # 反射越少越好
            score += reflection_score * weights['reflection']
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            print(f"活体评分计算失败: {e}")
            return 0.0
    
    def _identify_spoof_type(self, features: Dict[str, Any]) -> str:
        """识别欺骗类型"""
        try:
            # 基于特征判断欺骗类型
            reflection_ratio = features.get('reflection_ratio', 0)
            skin_ratio = features.get('skin_ratio', 0)
            edge_density = features.get('edge_density', 0)
            
            if reflection_ratio > 0.1:
                return "屏幕反射"
            elif skin_ratio < 0.3:
                return "非人脸物体"
            elif edge_density < 0.05:
                return "打印照片"
            else:
                return "未知"
        except Exception as e:
            print(f"欺骗类型识别失败: {e}")
            return "未知"
    
    def visualize_liveness_detection(self, image_path: str, save_path: str = None) -> np.ndarray:
        """
        可视化活体检测结果
        
        Args:
            image_path: 图像路径
            save_path: 保存路径（可选）
            
        Returns:
            可视化结果图像
        """
        try:
            img = cv2.imread(image_path)
            detection = self.detect_liveness(image_path)
            
            # 检测人脸
            faces = self.app.get(img)
            if len(faces) == 0:
                return img
            
            face = faces[0]
            bbox = face.bbox.astype(int)
            
            # 根据活体检测结果选择颜色
            if detection['is_live']:
                color = (0, 255, 0)  # 绿色 - 活体
                status = "活体"
            else:
                color = (0, 0, 255)  # 红色 - 非活体
                status = "非活体"
            
            # 绘制边界框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 绘制活体检测信息
            info_text = [
                f"状态: {status}",
                f"置信度: {detection['confidence']:.3f}",
                f"活体评分: {detection['liveness_score']:.3f}",
                f"欺骗类型: {detection['spoof_type']}"
            ]
            
            # 在图像上绘制文本
            y_offset = bbox[1] - 10
            for i, text in enumerate(info_text):
                cv2.putText(img, text, (bbox[0], y_offset - i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 保存结果
            if save_path:
                cv2.imwrite(save_path, img)
                print(f"活体检测结果已保存到: {save_path}")
            
            return img
            
        except Exception as e:
            print(f"可视化失败: {e}")
            return None


def demo_face_liveness():
    """人脸活体检测演示"""
    print("=== InsightFace 人脸活体检测演示 ===")
    
    # 创建活体检测器
    detector = FaceLivenessDetector()
    
    # 测试图像路径
    test_images = [
        "test_images/person1_1.jpg",
        "test_images/person2_1.jpg",
        "test_images/person2_2.jpg",
        "test_images/person2_3.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n检测图像: {img_path}")
            
            # 检测活体
            detection = detector.detect_liveness(img_path)
            
            print(f"活体状态: {'是' if detection['is_live'] else '否'}")
            print(f"置信度: {detection['confidence']:.3f}")
            print(f"活体评分: {detection['liveness_score']:.3f}")
            print(f"欺骗类型: {detection['spoof_type']}")
            
            # 显示详细特征
            features = detection.get('features', {})
            if features and 'error' not in features:
                print("\n详细特征:")
                print(f"  纹理熵: {features.get('lbp_entropy', 0):.3f}")
                print(f"  频率比: {features.get('freq_ratio', 0):.3f}")
                print(f"  边缘密度: {features.get('edge_density', 0):.3f}")
                print(f"  肤色比例: {features.get('skin_ratio', 0):.3f}")
                print(f"  深度评分: {features.get('depth_score', 0):.3f}")
                print(f"  反射比例: {features.get('reflection_ratio', 0):.3f}")
            
            # 可视化结果
            output_path = f"output/liveness_{os.path.basename(img_path)}"
            os.makedirs("output", exist_ok=True)
            detector.visualize_liveness_detection(img_path, output_path)
        else:
            print(f"图像不存在: {img_path}")


if __name__ == "__main__":
    demo_face_liveness()
