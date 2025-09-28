#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
苹果相册/Google相册风格的人脸聚类系统
实现高质量的人脸检测、特征提取和智能聚类功能
"""

import cv2
import numpy as np
import insightface
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os
import json
import sqlite3
from datetime import datetime
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import base64

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AppleStyleFaceClusterer:
    """苹果相册风格的人脸聚类器"""
    
    def __init__(self, model_name: str = 'buffalo_l', db_path: str = 'apple_style_clustering.db'):
        """
        初始化人脸聚类器
        
        Args:
            model_name: InsightFace模型名称
            db_path: 数据库路径
        """
        self.model_name = model_name
        self.db_path = db_path
        self.app = None
        self.face_detector = None
        self.face_recognizer = None
        
        # 聚类参数
        self.clustering_params = {
            'dbscan': {'eps': 0.35, 'min_samples': 2},
            'kmeans': {'n_clusters': 5},
            'hierarchical': {'n_clusters': 5, 'linkage': 'average'}
        }
        
        # 质量阈值
        self.quality_thresholds = {
            'min_face_size': 30,  # 最小人脸尺寸 (降低以包含更多人脸)
            'min_confidence': 0.65,  # 最小检测置信度 (降低以包含更多人脸)
            'min_quality_score': 0.25  # 最小质量分数 (降低以包含更多人脸)
        }
        
        self.load_model()
        self.init_database()
    
    def load_model(self):
        """加载InsightFace模型"""
        try:
            print("🔄 正在加载InsightFace模型...")
            self.app = insightface.app.FaceAnalysis(name=self.model_name)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print(f"✅ InsightFace模型 {self.model_name} 加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建人脸特征表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                face_id INTEGER,
                embedding BLOB NOT NULL,
                bbox TEXT,
                landmarks TEXT,
                confidence REAL,
                quality_score REAL,
                cluster_id INTEGER,
                person_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建聚类历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clustering_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                algorithm TEXT NOT NULL,
                parameters TEXT NOT NULL,
                total_faces INTEGER,
                total_clusters INTEGER,
                noise_faces INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建人物信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS person_info (
                cluster_id INTEGER PRIMARY KEY,
                person_name TEXT,
                representative_face_id INTEGER,
                face_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ 数据库初始化完成")
    
    def add_images_from_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        从目录添加图像进行人脸聚类
        
        Args:
            directory_path: 图像目录路径
            recursive: 是否递归搜索子目录
            
        Returns:
            添加结果统计
        """
        try:
            if not os.path.exists(directory_path):
                raise ValueError(f"目录不存在: {directory_path}")
            
            # 支持的图像格式
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            
            # 收集图像文件
            image_files = []
            if recursive:
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        if Path(file).suffix.lower() in image_extensions:
                            image_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(directory_path):
                    if Path(file).suffix.lower() in image_extensions:
                        image_files.append(os.path.join(directory_path, file))
            
            print(f"📸 发现 {len(image_files)} 个图像文件")
            
            # 处理图像
            results = {
                'total_images': len(image_files),
                'processed_images': 0,
                'total_faces': 0,
                'high_quality_faces': 0,
                'failed_images': 0,
                'errors': []
            }
            
            for i, img_path in enumerate(image_files):
                try:
                    print(f"🔄 处理图像 {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
                    
                    faces = self._extract_faces_from_image(img_path)
                    results['total_faces'] += len(faces)
                    
                    high_quality_count = 0
                    for face in faces:
                        if self._is_high_quality_face(face):
                            self._save_face_embedding(img_path, face)
                            high_quality_count += 1
                    
                    results['high_quality_faces'] += high_quality_count
                    results['processed_images'] += 1
                    
                    if high_quality_count > 0:
                        print(f"  ✅ 检测到 {len(faces)} 个人脸，{high_quality_count} 个高质量")
                    else:
                        print(f"  ⚠️  检测到 {len(faces)} 个人脸，但质量不足")
                        
                except Exception as e:
                    results['failed_images'] += 1
                    results['errors'].append(f"{img_path}: {str(e)}")
                    print(f"  ❌ 处理失败: {e}")
            
            print(f"\n📊 处理完成:")
            print(f"  总图像数: {results['total_images']}")
            print(f"  成功处理: {results['processed_images']}")
            print(f"  总人脸数: {results['total_faces']}")
            print(f"  高质量人脸: {results['high_quality_faces']}")
            print(f"  失败图像: {results['failed_images']}")
            
            return results
            
        except Exception as e:
            print(f"❌ 添加图像失败: {e}")
            return {'error': str(e)}
    
    def _extract_faces_from_image(self, image_path: str) -> List[Dict[str, Any]]:
        """从图像中提取人脸"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            # 使用InsightFace检测人脸
            faces = self.app.get(img)
            results = []
            
            for i, face in enumerate(faces):
                # 计算人脸质量分数
                quality_score = self._calculate_face_quality(img, face)
                
                # 提取关键点
                landmarks = face.kps.tolist() if hasattr(face, 'kps') else []
                
                results.append({
                    'face_id': i,
                    'embedding': face.embedding,
                    'bbox': face.bbox.tolist(),
                    'landmarks': landmarks,
                    'confidence': float(face.det_score),
                    'quality_score': quality_score,
                    'age': getattr(face, 'age', None),
                    'gender': getattr(face, 'gender', None)
                })
            
            return results
            
        except Exception as e:
            print(f"提取人脸失败 {image_path}: {e}")
            return []
    
    def _calculate_face_quality(self, img: np.ndarray, face) -> float:
        """计算人脸质量分数"""
        try:
            # 基础质量指标
            bbox = face.bbox
            x1, y1, x2, y2 = map(int, bbox)
            
            # 1. 人脸尺寸
            face_width = x2 - x1
            face_height = y2 - y1
            face_size_score = min(1.0, (face_width * face_height) / (100 * 100))
            
            # 2. 检测置信度
            confidence_score = float(face.det_score)
            
            # 3. 人脸区域清晰度（使用拉普拉斯算子）
            face_roi = img[y1:y2, x1:x2]
            if face_roi.size > 0:
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                sharpness_score = min(1.0, laplacian_var / 1000)
            else:
                sharpness_score = 0.0
            
            # 4. 人脸角度（基于关键点）
            angle_score = 1.0
            if hasattr(face, 'kps') and len(face.kps) >= 5:
                # 计算人脸角度
                left_eye = face.kps[0]
                right_eye = face.kps[1]
                nose = face.kps[2]
                
                # 计算眼睛连线角度
                eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
                angle_score = max(0.0, 1.0 - abs(eye_angle) / (np.pi / 4))
            
            # 综合质量分数
            quality_score = (
                face_size_score * 0.3 +
                confidence_score * 0.4 +
                sharpness_score * 0.2 +
                angle_score * 0.1
            )
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            print(f"计算人脸质量失败: {e}")
            return 0.0
    
    def _is_high_quality_face(self, face: Dict[str, Any]) -> bool:
        """判断是否为高质量人脸"""
        # 检查人脸尺寸
        bbox = face['bbox']
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        if (face_width < self.quality_thresholds['min_face_size'] or 
            face_height < self.quality_thresholds['min_face_size']):
            return False
        
        # 检查检测置信度
        if face['confidence'] < self.quality_thresholds['min_confidence']:
            return False
        
        # 检查质量分数
        if face['quality_score'] < self.quality_thresholds['min_quality_score']:
            return False
        
        return True
    
    def _save_face_embedding(self, image_path: str, face: Dict[str, Any]):
        """保存人脸特征到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO face_embeddings 
            (image_path, face_id, embedding, bbox, landmarks, confidence, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_path, 
            face['face_id'], 
            face['embedding'].tobytes(), 
            json.dumps(face['bbox']),
            json.dumps(face['landmarks']),
            face['confidence'],
            face['quality_score']
        ))
        
        conn.commit()
        conn.close()
    
    def load_all_embeddings(self) -> Tuple[List[int], np.ndarray, List[Dict[str, Any]]]:
        """加载所有特征向量"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, embedding, image_path, bbox, landmarks, confidence, quality_score, cluster_id
            FROM face_embeddings 
            ORDER BY id
        ''')
        
        ids = []
        embeddings = []
        metadata = []
        
        for row in cursor.fetchall():
            face_id, embedding_blob, image_path, bbox_json, landmarks_json, confidence, quality_score, cluster_id = row
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            
            ids.append(face_id)
            embeddings.append(embedding)
            metadata.append({
                'id': face_id,
                'image_path': image_path,
                'bbox': json.loads(bbox_json),
                'landmarks': json.loads(landmarks_json),
                'confidence': float(confidence),
                'quality_score': float(quality_score),
                'cluster_id': cluster_id
            })
        
        conn.close()
        return ids, np.array(embeddings), metadata
    
    def cluster_faces(self, algorithm: str = 'dbscan', **kwargs) -> Dict[str, Any]:
        """
        执行人脸聚类
        
        Args:
            algorithm: 聚类算法 ('dbscan', 'kmeans', 'hierarchical')
            **kwargs: 算法特定参数
            
        Returns:
            聚类结果
        """
        try:
            print(f"🔄 开始执行 {algorithm.upper()} 聚类...")
            
            ids, embeddings, metadata = self.load_all_embeddings()
            
            if len(embeddings) == 0:
                return {'clusters': [], 'total_faces': 0, 'total_clusters': 0, 'error': '没有可聚类的人脸'}
            
            print(f"📊 处理 {len(embeddings)} 个人脸特征")
            
            # 标准化特征向量
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # 执行聚类
            if algorithm.lower() == 'dbscan':
                params = {**self.clustering_params['dbscan'], **kwargs}
                clustering = DBSCAN(eps=params['eps'], min_samples=params['min_samples'], metric='cosine')
                cluster_labels = clustering.fit_predict(embeddings_scaled)
                
            elif algorithm.lower() == 'kmeans':
                params = {**self.clustering_params['kmeans'], **kwargs}
                clustering = KMeans(n_clusters=params['n_clusters'], random_state=42, n_init=10)
                cluster_labels = clustering.fit_predict(embeddings_scaled)
                
            elif algorithm.lower() == 'hierarchical':
                params = {**self.clustering_params['hierarchical'], **kwargs}
                clustering = AgglomerativeClustering(
                    n_clusters=params['n_clusters'], 
                    metric='cosine', 
                    linkage=params['linkage']
                )
                cluster_labels = clustering.fit_predict(embeddings_scaled)
                
            else:
                raise ValueError(f"不支持的聚类算法: {algorithm}")
            
            # 组织聚类结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                
                clusters[label].append({
                    'face_id': int(ids[i]),
                    'image_path': metadata[i]['image_path'],
                    'bbox': metadata[i]['bbox'],
                    'landmarks': metadata[i]['landmarks'],
                    'confidence': float(metadata[i]['confidence']),
                    'quality_score': float(metadata[i]['quality_score'])
                })
            
            # 更新数据库中的聚类标签
            self._update_cluster_labels(ids, cluster_labels)
            
            # 保存聚类历史
            self._save_clustering_history(algorithm, params, len(embeddings), len(clusters), 
                                        len([x for x in cluster_labels if x == -1]))
            
            # 转换格式
            cluster_list = []
            for cluster_id, faces in clusters.items():
                # 选择代表性人脸（质量最高的）
                representative_face = max(faces, key=lambda x: x['quality_score'])
                
                cluster_list.append({
                    'cluster_id': int(cluster_id),
                    'faces': faces,
                    'face_count': len(faces),
                    'representative_face': representative_face,
                    'avg_quality': np.mean([f['quality_score'] for f in faces])
                })
            
            # 按人脸数量排序
            cluster_list.sort(key=lambda x: x['face_count'], reverse=True)
            
            result = {
                'clusters': cluster_list,
                'total_faces': len(embeddings),
                'total_clusters': len(clusters),
                'noise_faces': len([x for x in cluster_labels if x == -1]),
                'algorithm': algorithm.upper(),
                'parameters': params,
                'success': True
            }
            
            print(f"✅ 聚类完成: {result['total_clusters']} 个聚类, {result['noise_faces']} 个噪声点")
            return result
            
        except Exception as e:
            print(f"❌ 聚类失败: {e}")
            return {'error': str(e), 'success': False}
    
    def _update_cluster_labels(self, face_ids: List[int], cluster_labels: List[int]):
        """更新数据库中的聚类标签"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for face_id, label in zip(face_ids, cluster_labels):
            cursor.execute('''
                UPDATE face_embeddings SET cluster_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
            ''', (int(label), face_id))
        
        conn.commit()
        conn.close()
    
    def _save_clustering_history(self, algorithm: str, parameters: Dict, total_faces: int, 
                               total_clusters: int, noise_faces: int):
        """保存聚类历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO clustering_history 
            (algorithm, parameters, total_faces, total_clusters, noise_faces)
            VALUES (?, ?, ?, ?, ?)
        ''', (algorithm, json.dumps(parameters), total_faces, total_clusters, noise_faces))
        
        conn.commit()
        conn.close()
    
    def visualize_clusters(self, save_path: str = None, max_faces_per_cluster: int = 9) -> None:
        """
        可视化聚类结果
        
        Args:
            save_path: 保存路径
            max_faces_per_cluster: 每个聚类最多显示的人脸数
        """
        try:
            print("🎨 生成聚类可视化...")
            
            # 获取聚类数据
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT cluster_id, COUNT(*) as face_count
                FROM face_embeddings 
                WHERE cluster_id IS NOT NULL AND cluster_id != -1
                GROUP BY cluster_id 
                ORDER BY face_count DESC
            ''')
            cluster_stats = cursor.fetchall()
            
            if not cluster_stats:
                print("❌ 没有聚类数据可视化")
                return
            
            # 创建可视化
            n_clusters = len(cluster_stats)
            cols = min(3, n_clusters)
            rows = (n_clusters + cols - 1) // cols
            
            fig = plt.figure(figsize=(cols * 6, rows * 6))
            fig.suptitle('苹果相册风格人脸聚类结果', fontsize=16, fontweight='bold')
            
            for i, (cluster_id, face_count) in enumerate(cluster_stats):
                ax = plt.subplot(rows, cols, i + 1)
                
                # 获取该聚类的人脸
                cursor.execute('''
                    SELECT image_path, bbox, quality_score
                    FROM face_embeddings 
                    WHERE cluster_id = ?
                    ORDER BY quality_score DESC
                    LIMIT ?
                ''', (cluster_id, max_faces_per_cluster))
                
                faces = cursor.fetchall()
                
                if not faces:
                    continue
                
                # 创建人脸网格
                face_size = 100
                grid_size = int(np.ceil(np.sqrt(len(faces))))
                cluster_img = np.ones((grid_size * face_size, grid_size * face_size, 3), dtype=np.uint8) * 255
                
                for j, (img_path, bbox_json, quality_score) in enumerate(faces):
                    if j >= max_faces_per_cluster:
                        break
                    
                    try:
                        # 读取图像
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        
                        # 提取人脸区域
                        bbox = json.loads(bbox_json)
                        x1, y1, x2, y2 = map(int, bbox)
                        face_img = img[y1:y2, x1:x2]
                        
                        if face_img.size == 0:
                            continue
                        
                        # 调整大小
                        face_img = cv2.resize(face_img, (face_size, face_size))
                        
                        # 放置到网格中
                        row = j // grid_size
                        col = j % grid_size
                        y_start = row * face_size
                        y_end = (row + 1) * face_size
                        x_start = col * face_size
                        x_end = (col + 1) * face_size
                        
                        cluster_img[y_start:y_end, x_start:x_end] = face_img
                        
                    except Exception as e:
                        print(f"处理人脸失败 {img_path}: {e}")
                        continue
                
                # 显示聚类
                ax.imshow(cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB))
                ax.set_title(f'聚类 {cluster_id}\n{face_count} 个人脸', fontsize=12, fontweight='bold')
                ax.axis('off')
            
            conn.close()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ 聚类可视化已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """获取聚类统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 总人脸数
            cursor.execute("SELECT COUNT(*) FROM face_embeddings")
            total_faces = cursor.fetchone()[0]
            
            # 聚类统计
            cursor.execute('''
                SELECT cluster_id, COUNT(*) as face_count, AVG(quality_score) as avg_quality
                FROM face_embeddings 
                WHERE cluster_id IS NOT NULL AND cluster_id != -1
                GROUP BY cluster_id 
                ORDER BY face_count DESC
            ''')
            cluster_stats = cursor.fetchall()
            
            # 噪声点数量
            cursor.execute("SELECT COUNT(*) FROM face_embeddings WHERE cluster_id = -1")
            noise_count = cursor.fetchone()[0]
            
            # 质量统计
            cursor.execute('''
                SELECT 
                    AVG(quality_score) as avg_quality,
                    MIN(quality_score) as min_quality,
                    MAX(quality_score) as max_quality,
                    AVG(confidence) as avg_confidence
                FROM face_embeddings
            ''')
            quality_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_faces': total_faces,
                'total_clusters': len(cluster_stats),
                'noise_faces': noise_count,
                'cluster_distribution': [
                    {
                        'cluster_id': cid, 
                        'face_count': count,
                        'avg_quality': float(avg_quality)
                    } 
                    for cid, count, avg_quality in cluster_stats
                ],
                'quality_stats': {
                    'avg_quality': float(quality_stats[0]) if quality_stats[0] else 0,
                    'min_quality': float(quality_stats[1]) if quality_stats[1] else 0,
                    'max_quality': float(quality_stats[2]) if quality_stats[2] else 0,
                    'avg_confidence': float(quality_stats[3]) if quality_stats[3] else 0
                }
            }
            
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {}
    
    def find_similar_faces(self, image_path: str, threshold: float = 0.6, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        查找相似人脸
        
        Args:
            image_path: 查询图像路径
            threshold: 相似度阈值
            max_results: 最大结果数
            
        Returns:
            相似人脸列表
        """
        try:
            # 提取查询图像的人脸特征
            faces = self._extract_faces_from_image(image_path)
            if not faces:
                return []
            
            query_face = max(faces, key=lambda x: x['quality_score'])
            query_embedding = query_face['embedding']
            
            # 加载所有特征向量
            ids, embeddings, metadata = self.load_all_embeddings()
            
            if len(embeddings) == 0:
                return []
            
            # 计算相似度
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            
            # 找到相似的人脸
            similar_faces = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    similar_faces.append({
                        'face_id': int(ids[i]),
                        'image_path': metadata[i]['image_path'],
                        'bbox': metadata[i]['bbox'],
                        'confidence': float(metadata[i]['confidence']),
                        'quality_score': float(metadata[i]['quality_score']),
                        'similarity': float(similarity),
                        'cluster_id': metadata[i]['cluster_id']
                    })
            
            # 按相似度排序并限制结果数
            similar_faces.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_faces[:max_results]
            
        except Exception as e:
            print(f"❌ 查找相似人脸失败: {e}")
            return []
    
    def export_clusters_to_json(self, output_path: str) -> bool:
        """导出聚类结果到JSON文件"""
        try:
            stats = self.get_cluster_statistics()
            
            # 获取详细的聚类数据
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT cluster_id, image_path, bbox, confidence, quality_score, created_at
                FROM face_embeddings 
                WHERE cluster_id IS NOT NULL AND cluster_id != -1
                ORDER BY cluster_id, quality_score DESC
            ''')
            
            clusters_data = {}
            for row in cursor.fetchall():
                cluster_id, image_path, bbox_json, confidence, quality_score, created_at = row
                
                if cluster_id not in clusters_data:
                    clusters_data[cluster_id] = {
                        'cluster_id': cluster_id,
                        'faces': []
                    }
                
                clusters_data[cluster_id]['faces'].append({
                    'image_path': image_path,
                    'bbox': json.loads(bbox_json),
                    'confidence': float(confidence),
                    'quality_score': float(quality_score),
                    'created_at': created_at
                })
            
            conn.close()
            
            # 组织导出数据
            export_data = {
                'export_info': {
                    'export_time': datetime.now().isoformat(),
                    'total_faces': stats['total_faces'],
                    'total_clusters': stats['total_clusters'],
                    'noise_faces': stats['noise_faces']
                },
                'statistics': stats,
                'clusters': list(clusters_data.values())
            }
            
            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 聚类结果已导出到: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            return False


def demo_apple_style_clustering():
    """苹果相册风格人脸聚类演示"""
    print("🍎 苹果相册风格人脸聚类演示")
    print("=" * 50)
    
    # 创建聚类器
    clusterer = AppleStyleFaceClusterer()
    
    # 测试图像目录
    test_dir = "../test_images"
    
    if os.path.exists(test_dir):
        print(f"\n📁 从目录 {test_dir} 添加图像...")
        results = clusterer.add_images_from_directory(test_dir, recursive=False)
        
        if results.get('high_quality_faces', 0) > 0:
            print(f"\n🔍 执行DBSCAN聚类...")
            dbscan_result = clusterer.cluster_faces('dbscan', eps=0.35, min_samples=2)
            
            if dbscan_result.get('success'):
                print(f"\n📊 聚类统计信息:")
                stats = clusterer.get_cluster_statistics()
                print(f"  总人脸数: {stats['total_faces']}")
                print(f"  聚类数: {stats['total_clusters']}")
                print(f"  噪声点: {stats['noise_faces']}")
                print(f"  平均质量: {stats['quality_stats']['avg_quality']:.3f}")
                
                print(f"\n📈 聚类分布:")
                for cluster in stats['cluster_distribution']:
                    print(f"  聚类 {cluster['cluster_id']}: {cluster['face_count']} 个人脸 (质量: {cluster['avg_quality']:.3f})")
                
                print(f"\n🎨 生成可视化...")
                os.makedirs("output", exist_ok=True)
                clusterer.visualize_clusters("output/apple_style_clusters.png")
                
                print(f"\n💾 导出结果...")
                clusterer.export_clusters_to_json("output/clusters_export.json")
                
            else:
                print(f"❌ 聚类失败: {dbscan_result.get('error', '未知错误')}")
        else:
            print("❌ 没有检测到高质量人脸")
    else:
        print(f"❌ 测试目录 {test_dir} 不存在，请添加测试图像")


if __name__ == "__main__":
    demo_apple_style_clustering()
