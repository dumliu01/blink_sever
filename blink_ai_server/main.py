"""
人脸识别聚类服务主程序
使用InsightFace实现人脸检测、特征提取和聚类功能
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import sqlite3
import numpy as np
from typing import List, Dict, Any
import cv2
from PIL import Image
import insightface
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import json
import base64
from io import BytesIO

app = FastAPI(title="人脸识别聚类服务", version="1.0.0")

# 全局变量
face_model = None
db_path = "face_clustering.db"

class FaceClusteringService:
    def __init__(self):
        self.face_model = None
        self.db_path = "face_clustering.db"
        self.init_database()
        self.load_model()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                face_id INTEGER,
                embedding BLOB NOT NULL,
                bbox TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def load_model(self):
        """加载InsightFace模型"""
        try:
            self.face_model = insightface.app.FaceAnalysis()
            self.face_model.prepare(ctx_id=0, det_size=(640, 640))
            print("✓ InsightFace模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("提示：如果是网络问题，请检查网络连接或使用代理")
            print("模型文件需要从GitHub下载，请确保网络畅通")
            # 不抛出异常，让服务继续运行但功能受限
            self.face_model = None
            print("⚠️  服务将在受限模式下运行（无法进行人脸检测）")
    
    def detect_and_extract_faces(self, image_path: str) -> List[Dict[str, Any]]:
        """检测并提取人脸特征"""
        if self.face_model is None:
            print("错误：InsightFace模型未加载，无法进行人脸检测")
            return []
            
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 检测人脸
            faces = self.face_model.get(img)
            
            results = []
            for i, face in enumerate(faces):
                # 提取特征向量
                embedding = face.embedding
                
                # 获取边界框
                bbox = face.bbox.tolist()
                
                # 获取置信度
                confidence = float(face.det_score)
                
                results.append({
                    'face_id': i,
                    'embedding': embedding.tolist(),  # 转换为Python列表
                    'bbox': bbox,
                    'confidence': confidence
                })
            
            return results
            
        except Exception as e:
            print(f"人脸检测失败: {e}")
            raise e
    
    def save_face_features(self, image_path: str, faces: List[Dict[str, Any]]):
        """保存人脸特征到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for face in faces:
            # 将numpy数组转换为二进制
            embedding_array = np.array(face['embedding'], dtype=np.float32)
            embedding_blob = embedding_array.tobytes()
            bbox_json = json.dumps(face['bbox'])
            
            cursor.execute('''
                INSERT INTO face_features (image_path, face_id, embedding, bbox, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_path, face['face_id'], embedding_blob, bbox_json, float(face['confidence'])))
        
        conn.commit()
        conn.close()
    
    def load_all_embeddings(self) -> tuple:
        """加载所有特征向量"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, embedding, image_path, bbox, confidence FROM face_features')
        
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
    
    def cluster_faces(self, eps: float = 0.4, min_samples: int = 2) -> Dict[str, Any]:
        """执行人脸聚类"""
        try:
            ids, embeddings, metadata = self.load_all_embeddings()
            
            if len(embeddings) == 0:
                return {'clusters': [], 'total_faces': 0, 'total_clusters': 0}
            
            # 使用DBSCAN进行聚类
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
            self.update_cluster_labels(ids, cluster_labels)
            
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
                'noise_faces': len([x for x in cluster_labels if x == -1])
            }
            
        except Exception as e:
            print(f"聚类失败: {e}")
            raise e
    
    def update_cluster_labels(self, face_ids: List[int], cluster_labels: List[int]):
        """更新数据库中的聚类标签"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 添加cluster_id列（如果不存在）
        try:
            cursor.execute('ALTER TABLE face_features ADD COLUMN cluster_id INTEGER')
        except sqlite3.OperationalError:
            pass  # 列已存在
        
        for face_id, label in zip(face_ids, cluster_labels):
            cursor.execute('''
                UPDATE face_features SET cluster_id = ? WHERE id = ?
            ''', (int(label), face_id))
        
        conn.commit()
        conn.close()
    
    def find_similar_faces(self, image_path: str, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """查找相似人脸"""
        try:
            # 提取查询图像的人脸特征
            faces = self.detect_and_extract_faces(image_path)
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
            raise e

# 初始化服务
face_service = FaceClusteringService()

@app.get("/")
async def root():
    return {"message": "人脸识别聚类服务运行中"}

@app.post("/detect_faces")
async def detect_faces(file: UploadFile = File(...)):
    """检测并提取人脸特征"""
    try:
        # 保存上传的文件
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 检测人脸
        faces = face_service.detect_and_extract_faces(file_path)
        
        # 保存特征
        face_service.save_face_features(file_path, faces)
        
        return {
            "message": "人脸检测完成",
            "image_path": file_path,
            "face_count": len(faces),
            "faces": [
                {
                    "face_id": int(face['face_id']),
                    "bbox": face['bbox'],
                    "confidence": float(face['confidence'])
                }
                for face in faces
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cluster_faces")
async def cluster_faces(eps: float = 0.4, min_samples: int = 2):
    """执行人脸聚类"""
    try:
        result = face_service.cluster_faces(eps, min_samples)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find_similar")
async def find_similar_faces(file: UploadFile = File(...), threshold: float = 0.6):
    """查找相似人脸"""
    try:
        # 保存上传的文件
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, f"query_{file.filename}")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 查找相似人脸
        similar_faces = face_service.find_similar_faces(file_path, threshold)
        
        return {
            "message": "相似人脸查找完成",
            "query_image": file_path,
            "similar_faces": similar_faces,
            "count": len(similar_faces)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clusters")
async def get_clusters():
    """获取所有聚类结果"""
    try:
        result = face_service.cluster_faces()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    try:
        conn = sqlite3.connect(face_service.db_path)
        cursor = conn.cursor()
        
        # 总人脸数
        cursor.execute("SELECT COUNT(*) FROM face_features")
        total_faces = cursor.fetchone()[0]
        
        # 聚类统计
        cursor.execute("SELECT cluster_id, COUNT(*) FROM face_features WHERE cluster_id IS NOT NULL GROUP BY cluster_id")
        cluster_stats = cursor.fetchall()
        
        # 噪声点数量
        cursor.execute("SELECT COUNT(*) FROM face_features WHERE cluster_id = -1 OR cluster_id IS NULL")
        noise_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_faces": total_faces,
            "total_clusters": len(cluster_stats),
            "noise_faces": noise_count,
            "cluster_distribution": [{"cluster_id": cid, "face_count": count} for cid, count in cluster_stats]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
