"""
InsightFace 人脸聚类演示
包含人脸特征聚类、相似人脸分组等功能
"""

import cv2
import numpy as np
import insightface
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import json
import sqlite3
from datetime import datetime


class FaceClusterer:
    """人脸聚类器类"""
    
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        初始化人脸聚类器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.app = None
        self.db_path = "face_clustering.db"
        self.load_model()
        self.init_database()
    
    def load_model(self):
        """加载InsightFace模型"""
        try:
            self.app = insightface.app.FaceAnalysis(name=self.model_name)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print(f"✓ InsightFace模型 {self.model_name} 加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e
    
    def init_database(self):
        """初始化聚类数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                face_id INTEGER,
                embedding BLOB NOT NULL,
                bbox TEXT,
                confidence REAL,
                cluster_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_faces_from_directory(self, directory_path: str) -> int:
        """
        从目录添加人脸图像进行聚类
        
        Args:
            directory_path: 图像目录路径
            
        Returns:
            添加的人脸数量
        """
        try:
            if not os.path.exists(directory_path):
                raise ValueError(f"目录不存在: {directory_path}")
            
            added_count = 0
            image_files = [f for f in os.listdir(directory_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"发现 {len(image_files)} 个图像文件")
            
            for img_file in image_files:
                img_path = os.path.join(directory_path, img_file)
                faces = self._extract_faces_from_image(img_path)
                
                for face in faces:
                    self._save_face_embedding(img_path, face)
                    added_count += 1
            
            print(f"✓ 成功添加 {added_count} 个人脸")
            return added_count
            
        except Exception as e:
            print(f"添加人脸失败: {e}")
            return 0
    
    def _extract_faces_from_image(self, image_path: str) -> List[Dict[str, Any]]:
        """从图像中提取人脸"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            faces = self.app.get(img)
            results = []
            
            for i, face in enumerate(faces):
                results.append({
                    'face_id': i,
                    'embedding': face.embedding,
                    'bbox': face.bbox.tolist(),
                    'confidence': float(face.det_score)
                })
            
            return results
            
        except Exception as e:
            print(f"提取人脸失败 {image_path}: {e}")
            return []
    
    def _save_face_embedding(self, image_path: str, face: Dict[str, Any]):
        """保存人脸特征到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO face_embeddings (image_path, face_id, embedding, bbox, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (image_path, face['face_id'], face['embedding'].tobytes(), 
              json.dumps(face['bbox']), face['confidence']))
        
        conn.commit()
        conn.close()
    
    def load_all_embeddings(self) -> Tuple[List[int], np.ndarray, List[Dict[str, Any]]]:
        """加载所有特征向量"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, embedding, image_path, bbox, confidence 
            FROM face_embeddings 
            ORDER BY id
        ''')
        
        ids = []
        embeddings = []
        metadata = []
        
        for row in cursor.fetchall():
            face_id, embedding_blob, image_path, bbox_json, confidence = row
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            
            ids.append(face_id)
            embeddings.append(embedding)
            metadata.append({
                'id': face_id,
                'image_path': image_path,
                'bbox': json.loads(bbox_json),
                'confidence': float(confidence)
            })
        
        conn.close()
        return ids, np.array(embeddings), metadata
    
    def cluster_faces_dbscan(self, eps: float = 0.4, min_samples: int = 2) -> Dict[str, Any]:
        """
        使用DBSCAN进行人脸聚类
        
        Args:
            eps: 邻域半径
            min_samples: 最小样本数
            
        Returns:
            聚类结果
        """
        try:
            ids, embeddings, metadata = self.load_all_embeddings()
            
            if len(embeddings) == 0:
                return {'clusters': [], 'total_faces': 0, 'total_clusters': 0}
            
            # 使用DBSCAN聚类
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            cluster_labels = clustering.fit_predict(embeddings)
            
            # 组织聚类结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                
                clusters[label].append({
                    'face_id': int(ids[i]),
                    'image_path': metadata[i]['image_path'],
                    'bbox': metadata[i]['bbox'],
                    'confidence': float(metadata[i]['confidence'])
                })
            
            # 更新数据库中的聚类标签
            self._update_cluster_labels(ids, cluster_labels)
            
            # 转换格式
            cluster_list = []
            for cluster_id, faces in clusters.items():
                cluster_list.append({
                    'cluster_id': int(cluster_id),
                    'faces': faces,
                    'face_count': len(faces)
                })
            
            return {
                'clusters': cluster_list,
                'total_faces': len(embeddings),
                'total_clusters': len(clusters),
                'noise_faces': len([x for x in cluster_labels if x == -1]),
                'algorithm': 'DBSCAN',
                'parameters': {'eps': eps, 'min_samples': min_samples}
            }
            
        except Exception as e:
            print(f"DBSCAN聚类失败: {e}")
            raise e
    
    def cluster_faces_kmeans(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        使用K-Means进行人脸聚类
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            聚类结果
        """
        try:
            ids, embeddings, metadata = self.load_all_embeddings()
            
            if len(embeddings) == 0:
                return {'clusters': [], 'total_faces': 0, 'total_clusters': 0}
            
            # 使用K-Means聚类
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clustering.fit_predict(embeddings)
            
            # 组织聚类结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                
                clusters[label].append({
                    'face_id': int(ids[i]),
                    'image_path': metadata[i]['image_path'],
                    'bbox': metadata[i]['bbox'],
                    'confidence': float(metadata[i]['confidence'])
                })
            
            # 更新数据库中的聚类标签
            self._update_cluster_labels(ids, cluster_labels)
            
            # 转换格式
            cluster_list = []
            for cluster_id, faces in clusters.items():
                cluster_list.append({
                    'cluster_id': int(cluster_id),
                    'faces': faces,
                    'face_count': len(faces)
                })
            
            return {
                'clusters': cluster_list,
                'total_faces': len(embeddings),
                'total_clusters': len(clusters),
                'noise_faces': 0,
                'algorithm': 'K-Means',
                'parameters': {'n_clusters': n_clusters}
            }
            
        except Exception as e:
            print(f"K-Means聚类失败: {e}")
            raise e
    
    def cluster_faces_hierarchical(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        使用层次聚类进行人脸聚类
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            聚类结果
        """
        try:
            ids, embeddings, metadata = self.load_all_embeddings()
            
            if len(embeddings) == 0:
                return {'clusters': [], 'total_faces': 0, 'total_clusters': 0}
            
            # 使用层次聚类
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
            cluster_labels = clustering.fit_predict(embeddings)
            
            # 组织聚类结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                
                clusters[label].append({
                    'face_id': int(ids[i]),
                    'image_path': metadata[i]['image_path'],
                    'bbox': metadata[i]['bbox'],
                    'confidence': float(metadata[i]['confidence'])
                })
            
            # 更新数据库中的聚类标签
            self._update_cluster_labels(ids, cluster_labels)
            
            # 转换格式
            cluster_list = []
            for cluster_id, faces in clusters.items():
                cluster_list.append({
                    'cluster_id': int(cluster_id),
                    'faces': faces,
                    'face_count': len(faces)
                })
            
            return {
                'clusters': cluster_list,
                'total_faces': len(embeddings),
                'total_clusters': len(clusters),
                'noise_faces': 0,
                'algorithm': 'Hierarchical',
                'parameters': {'n_clusters': n_clusters}
            }
            
        except Exception as e:
            print(f"层次聚类失败: {e}")
            raise e
    
    def _update_cluster_labels(self, face_ids: List[int], cluster_labels: List[int]):
        """更新数据库中的聚类标签"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for face_id, label in zip(face_ids, cluster_labels):
            cursor.execute('''
                UPDATE face_embeddings SET cluster_id = ? WHERE id = ?
            ''', (int(label), face_id))
        
        conn.commit()
        conn.close()
    
    def find_similar_faces(self, image_path: str, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        查找相似人脸
        
        Args:
            image_path: 查询图像路径
            threshold: 相似度阈值
            
        Returns:
            相似人脸列表
        """
        try:
            # 提取查询图像的人脸特征
            faces = self._extract_faces_from_image(image_path)
            if not faces:
                return []
            
            query_embedding = faces[0]['embedding']
            
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
                        'similarity': float(similarity)
                    })
            
            # 按相似度排序
            similar_faces.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similar_faces
            
        except Exception as e:
            print(f"查找相似人脸失败: {e}")
            return []
    
    def visualize_clusters(self, save_path: str = None) -> None:
        """
        可视化聚类结果
        
        Args:
            save_path: 保存路径
        """
        try:
            ids, embeddings, metadata = self.load_all_embeddings()
            
            if len(embeddings) == 0:
                print("没有数据可视化")
                return
            
            # 使用PCA降维到2D
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # 获取聚类标签
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT cluster_id FROM face_embeddings ORDER BY id')
            cluster_labels = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            # 创建可视化
            plt.figure(figsize=(12, 8))
            
            # 绘制聚类结果
            unique_labels = set(cluster_labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                if label == -1:  # 噪声点
                    plt.scatter(embeddings_2d[cluster_labels == label, 0],
                              embeddings_2d[cluster_labels == label, 1],
                              c='black', marker='x', s=50, alpha=0.6, label='噪声点')
                else:
                    plt.scatter(embeddings_2d[cluster_labels == label, 0],
                              embeddings_2d[cluster_labels == label, 1],
                              c=[colors[i]], s=50, alpha=0.7, label=f'聚类 {label}')
            
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('人脸聚类可视化 (PCA降维)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"聚类可视化已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"可视化失败: {e}")
    
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
                SELECT cluster_id, COUNT(*) 
                FROM face_embeddings 
                WHERE cluster_id IS NOT NULL 
                GROUP BY cluster_id 
                ORDER BY cluster_id
            ''')
            cluster_stats = cursor.fetchall()
            
            # 噪声点数量
            cursor.execute("SELECT COUNT(*) FROM face_embeddings WHERE cluster_id = -1")
            noise_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_faces': total_faces,
                'total_clusters': len(cluster_stats),
                'noise_faces': noise_count,
                'cluster_distribution': [
                    {'cluster_id': cid, 'face_count': count} 
                    for cid, count in cluster_stats
                ]
            }
            
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {}


def demo_face_clustering():
    """人脸聚类演示"""
    print("=== InsightFace 人脸聚类演示 ===")
    
    # 创建聚类器
    clusterer = FaceClusterer()
    
    # 测试图像目录
    test_dir = "test_images"
    
    if os.path.exists(test_dir):
        print(f"\n1. 从目录 {test_dir} 添加人脸...")
        added_count = clusterer.add_faces_from_directory(test_dir)
        print(f"添加了 {added_count} 个人脸")
        
        if added_count > 0:
            print("\n2. 执行DBSCAN聚类...")
            dbscan_result = clusterer.cluster_faces_dbscan(eps=0.4, min_samples=2)
            print(f"DBSCAN结果: {dbscan_result['total_clusters']} 个聚类, {dbscan_result['noise_faces']} 个噪声点")
            
            print("\n3. 执行K-Means聚类...")
            kmeans_result = clusterer.cluster_faces_kmeans(n_clusters=3)
            print(f"K-Means结果: {kmeans_result['total_clusters']} 个聚类")
            
            print("\n4. 聚类统计信息:")
            stats = clusterer.get_cluster_statistics()
            print(f"  总人脸数: {stats['total_faces']}")
            print(f"  聚类数: {stats['total_clusters']}")
            print(f"  噪声点: {stats['noise_faces']}")
            
            print("\n5. 聚类分布:")
            for cluster in stats['cluster_distribution']:
                print(f"  聚类 {cluster['cluster_id']}: {cluster['face_count']} 个人脸")
            
            print("\n6. 可视化聚类结果...")
            os.makedirs("output", exist_ok=True)
            clusterer.visualize_clusters("output/cluster_visualization.png")
            
    else:
        print(f"测试目录 {test_dir} 不存在，请添加测试图像")


if __name__ == "__main__":
    demo_face_clustering()
