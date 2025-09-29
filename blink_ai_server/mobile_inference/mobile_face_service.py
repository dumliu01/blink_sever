"""
移动端人脸识别服务
提供量化模型的人脸检测和识别功能
"""

import os
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import base64
from io import BytesIO
from .onnx_inference import ONNXInference

logger = logging.getLogger(__name__)

class MobileFaceService:
    """移动端人脸识别服务类"""
    
    def __init__(self, detection_model_path: str, recognition_model_path: str,
                 providers: Optional[List[str]] = None):
        """
        初始化移动端人脸识别服务
        
        Args:
            detection_model_path: 人脸检测模型路径
            recognition_model_path: 人脸识别模型路径
            providers: 推理提供者列表
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            # 初始化检测模型
            self.detection_engine = ONNXInference(detection_model_path, providers)
            
            # 初始化识别模型
            self.recognition_engine = ONNXInference(recognition_model_path, providers)
            
            self.logger.info("移动端人脸识别服务初始化成功")
            
        except Exception as e:
            self.logger.error(f"移动端人脸识别服务初始化失败: {e}")
            raise e
    
    def detect_faces(self, image: Union[str, np.ndarray, bytes], 
                    confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        检测人脸
        
        Args:
            image: 图像（路径、numpy数组或base64编码的字节）
            confidence_threshold: 置信度阈值
            
        Returns:
            List[Dict]: 检测到的人脸信息列表
        """
        try:
            # 预处理图像
            processed_image = self.detection_engine.preprocess_image(image)
            
            # 执行检测推理
            outputs = self.detection_engine.predict(processed_image)
            
            # 解析检测结果
            faces = self._parse_detection_outputs(outputs, confidence_threshold)
            
            self.logger.info(f"检测到 {len(faces)} 个人脸")
            return faces
            
        except Exception as e:
            self.logger.error(f"人脸检测失败: {e}")
            return []
    
    def extract_face_embedding(self, image: Union[str, np.ndarray, bytes], 
                              bbox: Optional[List[float]] = None) -> Optional[np.ndarray]:
        """
        提取人脸特征向量
        
        Args:
            image: 图像（路径、numpy数组或base64编码的字节）
            bbox: 人脸边界框，如果为None则使用整张图像
            
        Returns:
            Optional[np.ndarray]: 人脸特征向量
        """
        try:
            # 如果提供了边界框，先裁剪人脸区域
            if bbox is not None:
                image = self._crop_face_region(image, bbox)
            
            # 预处理图像
            processed_image = self.recognition_engine.preprocess_image(image)
            
            # 执行识别推理
            outputs = self.recognition_engine.predict(processed_image)
            
            # 提取特征向量
            if outputs and len(outputs) > 0:
                embedding = outputs[0].flatten()
                # 归一化特征向量
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            
            return None
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            return None
    
    def recognize_faces(self, image: Union[str, np.ndarray, bytes], 
                       known_embeddings: List[np.ndarray],
                       known_labels: List[str],
                       similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        识别人脸
        
        Args:
            image: 图像（路径、numpy数组或base64编码的字节）
            known_embeddings: 已知人脸特征向量列表
            known_labels: 已知人脸标签列表
            similarity_threshold: 相似度阈值
            
        Returns:
            List[Dict]: 识别结果列表
        """
        try:
            # 检测人脸
            faces = self.detect_faces(image)
            
            if not faces:
                return []
            
            # 对每个检测到的人脸进行识别
            results = []
            for face in faces:
                bbox = face['bbox']
                confidence = face['confidence']
                
                # 提取特征向量
                embedding = self.extract_face_embedding(image, bbox)
                
                if embedding is not None:
                    # 计算与已知人脸的相似度
                    similarities = []
                    for known_embedding in known_embeddings:
                        similarity = np.dot(embedding, known_embedding)
                        similarities.append(similarity)
                    
                    # 找到最相似的人脸
                    max_similarity = max(similarities) if similarities else 0
                    best_match_idx = similarities.index(max_similarity) if similarities else -1
                    
                    if max_similarity >= similarity_threshold and best_match_idx >= 0:
                        result = {
                            'bbox': bbox,
                            'confidence': confidence,
                            'label': known_labels[best_match_idx],
                            'similarity': float(max_similarity),
                            'embedding': embedding.tolist()
                        }
                    else:
                        result = {
                            'bbox': bbox,
                            'confidence': confidence,
                            'label': 'unknown',
                            'similarity': float(max_similarity),
                            'embedding': embedding.tolist()
                        }
                    
                    results.append(result)
            
            self.logger.info(f"识别完成，找到 {len(results)} 个有效人脸")
            return results
            
        except Exception as e:
            self.logger.error(f"人脸识别失败: {e}")
            return []
    
    def _parse_detection_outputs(self, outputs: List[np.ndarray], 
                                confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        解析检测模型输出
        
        Args:
            outputs: 模型输出
            confidence_threshold: 置信度阈值
            
        Returns:
            List[Dict]: 解析后的人脸信息
        """
        try:
            faces = []
            
            if len(outputs) >= 2:
                boxes = outputs[0]  # 边界框
                scores = outputs[1]  # 置信度
                
                # 过滤低置信度的检测结果
                valid_indices = scores > confidence_threshold
                valid_boxes = boxes[valid_indices]
                valid_scores = scores[valid_indices]
                
                for i, (box, score) in enumerate(zip(valid_boxes, valid_scores)):
                    # 假设边界框格式为 [x1, y1, x2, y2]
                    if len(box) >= 4:
                        face_info = {
                            'face_id': i,
                            'bbox': box[:4].tolist(),
                            'confidence': float(score)
                        }
                        faces.append(face_info)
            
            return faces
            
        except Exception as e:
            self.logger.error(f"解析检测输出失败: {e}")
            return []
    
    def _crop_face_region(self, image: Union[str, np.ndarray, bytes], 
                         bbox: List[float]) -> np.ndarray:
        """
        裁剪人脸区域
        
        Args:
            image: 原始图像
            bbox: 边界框 [x1, y1, x2, y2]
            
        Returns:
            np.ndarray: 裁剪后的人脸图像
        """
        try:
            # 加载图像
            if isinstance(image, str):
                if os.path.exists(image):
                    img = cv2.imread(image)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    # base64字符串
                    img_data = base64.b64decode(image)
                    img = Image.open(BytesIO(img_data))
                    img = np.array(img)
            elif isinstance(image, bytes):
                img = Image.open(BytesIO(image))
                img = np.array(img)
            elif isinstance(image, np.ndarray):
                img = image
            else:
                raise ValueError("不支持的图像格式")
            
            # 确保是RGB格式
            if len(img.shape) == 3 and img.shape[2] == 3:
                pass
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            else:
                raise ValueError("图像必须是3通道RGB格式")
            
            # 裁剪人脸区域
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            
            face_crop = img[y1:y2, x1:x2]
            
            return face_crop
            
        except Exception as e:
            self.logger.error(f"裁剪人脸区域失败: {e}")
            return image if isinstance(image, np.ndarray) else np.array([])
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        获取服务信息
        
        Returns:
            Dict: 服务信息
        """
        try:
            detection_info = self.detection_engine.get_model_info()
            recognition_info = self.recognition_engine.get_model_info()
            
            return {
                "detection_model": detection_info,
                "recognition_model": recognition_info,
                "service_status": "running"
            }
            
        except Exception as e:
            self.logger.error(f"获取服务信息失败: {e}")
            return {"service_status": "error", "error": str(e)}
