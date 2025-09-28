"""
InsightFace 人脸识别演示
包含人脸特征提取、人脸验证、人脸识别等功能
"""

import cv2
import numpy as np
import insightface
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity
import json
import sqlite3
from datetime import datetime


class FaceRecognizer:
    """人脸识别器类"""
    
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        初始化人脸识别器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.app = None
        self.db_path = "face_database.db"
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
        """初始化人脸数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                person_name TEXT,
                image_path TEXT,
                embedding BLOB NOT NULL,
                bbox TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_embedding(self, image_path: str, face_index: int = 0) -> np.ndarray:
        """
        提取人脸特征向量
        
        Args:
            image_path: 图像路径
            face_index: 人脸索引
            
        Returns:
            512维特征向量
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            faces = self.app.get(img)
            if face_index >= len(faces):
                raise ValueError(f"人脸索引 {face_index} 超出范围")
            
            face = faces[face_index]
            return face.embedding
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            raise e
    
    def register_person(self, person_id: str, person_name: str, image_path: str) -> bool:
        """
        注册新人员
        
        Args:
            person_id: 人员ID
            person_name: 人员姓名
            image_path: 图像路径
            
        Returns:
            是否注册成功
        """
        try:
            # 检测人脸
            img = cv2.imread(image_path)
            faces = self.app.get(img)
            
            if len(faces) == 0:
                print("未检测到人脸")
                return False
            
            # 使用第一个检测到的人脸
            face = faces[0]
            embedding = face.embedding
            bbox = face.bbox.tolist()
            confidence = float(face.det_score)
            
            # 保存到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO face_embeddings (person_id, person_name, image_path, embedding, bbox, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (person_id, person_name, image_path, embedding.tobytes(), 
                  json.dumps(bbox), confidence))
            
            conn.commit()
            conn.close()
            
            print(f"✓ 人员 {person_name} 注册成功")
            return True
            
        except Exception as e:
            print(f"人员注册失败: {e}")
            return False
    
    def verify_face(self, image_path: str, person_id: str, threshold: float = 0.6) -> Dict[str, Any]:
        """
        人脸验证
        
        Args:
            image_path: 待验证图像路径
            person_id: 目标人员ID
            threshold: 相似度阈值
            
        Returns:
            验证结果
        """
        try:
            # 提取待验证图像的特征
            query_embedding = self.extract_embedding(image_path)
            
            # 从数据库获取目标人员的所有特征
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT embedding, person_name FROM face_embeddings 
                WHERE person_id = ?
            ''', (person_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {
                    'verified': False,
                    'similarity': 0.0,
                    'message': '目标人员未注册'
                }
            
            # 计算与所有注册图像的最大相似度
            max_similarity = 0.0
            for embedding_blob, person_name in results:
                stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
                max_similarity = max(max_similarity, similarity)
            
            verified = max_similarity >= threshold
            
            return {
                'verified': verified,
                'similarity': float(max_similarity),
                'threshold': threshold,
                'person_name': results[0][1] if results else 'Unknown',
                'message': '验证通过' if verified else '验证失败'
            }
            
        except Exception as e:
            print(f"人脸验证失败: {e}")
            return {
                'verified': False,
                'similarity': 0.0,
                'message': f'验证错误: {str(e)}'
            }
    
    def identify_face(self, image_path: str, threshold: float = 0.6) -> Dict[str, Any]:
        """
        人脸识别
        
        Args:
            image_path: 待识别图像路径
            threshold: 相似度阈值
            
        Returns:
            识别结果
        """
        try:
            # 提取待识别图像的特征
            query_embedding = self.extract_embedding(image_path)
            
            # 从数据库获取所有注册人员的特征
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT person_id, person_name, embedding FROM face_embeddings
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {
                    'identified': False,
                    'person_id': None,
                    'person_name': None,
                    'similarity': 0.0,
                    'message': '数据库为空'
                }
            
            # 计算与所有注册人员的相似度
            best_match = None
            max_similarity = 0.0
            
            for person_id, person_name, embedding_blob in results:
                stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = {
                        'person_id': person_id,
                        'person_name': person_name,
                        'similarity': similarity
                    }
            
            identified = max_similarity >= threshold
            
            return {
                'identified': identified,
                'person_id': best_match['person_id'] if identified else None,
                'person_name': best_match['person_name'] if identified else None,
                'similarity': float(max_similarity),
                'threshold': threshold,
                'message': f"识别为 {best_match['person_name']}" if identified else '未识别'
            }
            
        except Exception as e:
            print(f"人脸识别失败: {e}")
            return {
                'identified': False,
                'person_id': None,
                'person_name': None,
                'similarity': 0.0,
                'message': f'识别错误: {str(e)}'
            }
    
    def get_all_persons(self) -> List[Dict[str, Any]]:
        """获取所有注册人员信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT person_id, person_name, COUNT(*) as image_count, 
                       MAX(created_at) as last_updated
                FROM face_embeddings 
                GROUP BY person_id, person_name
                ORDER BY last_updated DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            persons = []
            for person_id, person_name, image_count, last_updated in results:
                persons.append({
                    'person_id': person_id,
                    'person_name': person_name,
                    'image_count': image_count,
                    'last_updated': last_updated
                })
            
            return persons
            
        except Exception as e:
            print(f"获取人员信息失败: {e}")
            return []
    
    def delete_person(self, person_id: str) -> bool:
        """删除人员"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM face_embeddings WHERE person_id = ?', (person_id,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                print(f"✓ 已删除人员 {person_id}")
                return True
            else:
                print(f"未找到人员 {person_id}")
                return False
                
        except Exception as e:
            print(f"删除人员失败: {e}")
            return False


def demo_face_recognition():
    """人脸识别演示"""
    print("=== InsightFace 人脸识别演示 ===")
    
    # 创建识别器
    recognizer = FaceRecognizer()
    
    # 测试图像路径
    test_images = {
        "person1": "test_images/person1_1.jpg",
        "person2": "test_images/person2_1.jpg",
        "person2_2": "test_images/person2_2.jpg",
        "person2_3": "test_images/person2_3.jpg"
    }
    
    # 注册人员
    print("\n1. 注册人员...")
    for person_id, img_path in test_images.items():
        if os.path.exists(img_path):
            person_name = f"人员{person_id}"
            recognizer.register_person(person_id, person_name, img_path)
    
    # 显示注册的人员
    print("\n2. 已注册人员:")
    persons = recognizer.get_all_persons()
    for person in persons:
        print(f"  {person['person_name']} (ID: {person['person_id']}) - {person['image_count']}张图像")
    
    # 人脸识别测试
    print("\n3. 人脸识别测试...")
    for person_id, img_path in test_images.items():
        if os.path.exists(img_path):
            print(f"\n识别图像: {img_path}")
            result = recognizer.identify_face(img_path)
            print(f"  结果: {result['message']}")
            print(f"  相似度: {result['similarity']:.3f}")
            if result['identified']:
                print(f"  识别为: {result['person_name']}")
    
    # 人脸验证测试
    print("\n4. 人脸验证测试...")
    if persons:
        test_person = persons[0]
        print(f"验证目标: {test_person['person_name']}")
        
        for person_id, img_path in test_images.items():
            if os.path.exists(img_path):
                print(f"\n验证图像: {img_path}")
                result = recognizer.verify_face(img_path, test_person['person_id'])
                print(f"  结果: {result['message']}")
                print(f"  相似度: {result['similarity']:.3f}")


if __name__ == "__main__":
    demo_face_recognition()
