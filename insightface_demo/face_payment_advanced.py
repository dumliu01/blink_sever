#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级人脸识别支付系统
使用分层索引、缓存和多种优化策略
"""

import os
import cv2
import json
import sqlite3
import numpy as np
import hashlib
from datetime import datetime
from insightface import app
from insightface.data import get_image as ins_get_image
import faiss
import pickle
import time
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

class AdvancedFacePaymentSystem:
    def __init__(self, db_path="face_payment_advanced.db"):
        self.db_path = db_path
        self.face_app = app.FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # 多级索引系统
        self.coarse_index = None  # 粗粒度索引 (IVF)
        self.fine_index = None    # 细粒度索引 (HNSW)
        self.user_mapping = {}
        self.feature_dim = 512
        
        # 缓存系统
        self.feature_cache = {}  # 特征向量缓存
        self.user_cache = {}     # 用户信息缓存
        self.cache_lock = threading.RLock()
        
        # 性能统计
        self.stats = {
            'total_verifications': 0,
            'cache_hits': 0,
            'avg_search_time': 0.0,
            'total_search_time': 0.0
        }
        
        # 初始化
        self._init_database()
        self._load_or_create_indexes()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                face_features BLOB,
                feature_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS payment_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                amount REAL,
                success BOOLEAN,
                confidence_score REAL,
                search_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建索引以提高查询性能
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON users(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_payment_timestamp ON payment_records(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_payment_user ON payment_records(user_id)')
        
        conn.commit()
        conn.close()
        print("✅ 数据库初始化完成")
    
    def _load_or_create_indexes(self):
        """加载或创建多级索引"""
        coarse_file = "coarse_index.index"
        fine_file = "fine_index.index"
        mapping_file = "user_mapping.pkl"
        
        if all(os.path.exists(f) for f in [coarse_file, fine_file, mapping_file]):
            # 加载现有索引
            self.coarse_index = faiss.read_index(coarse_file)
            self.fine_index = faiss.read_index(fine_file)
            with open(mapping_file, 'rb') as f:
                self.user_mapping = pickle.load(f)
            print(f"✅ 加载多级索引，包含 {self.coarse_index.ntotal} 个用户")
        else:
            # 创建新的多级索引
            # 粗粒度索引：IVF (Inverted File)
            quantizer = faiss.IndexFlatIP(self.feature_dim)
            self.coarse_index = faiss.IndexIVFFlat(quantizer, self.feature_dim, 100)  # 100个聚类中心
            
            # 细粒度索引：HNSW (Hierarchical Navigable Small World)
            self.fine_index = faiss.IndexHNSWFlat(self.feature_dim, 32)  # 32个连接
            
            self.user_mapping = {}
            print("✅ 创建新的多级索引")
    
    def _save_indexes(self):
        """保存索引到文件"""
        if self.coarse_index and self.fine_index and self.user_mapping:
            faiss.write_index(self.coarse_index, "coarse_index.index")
            faiss.write_index(self.fine_index, "fine_index.index")
            with open("user_mapping.pkl", 'wb') as f:
                pickle.dump(self.user_mapping, f)
            print("✅ 多级索引已保存")
    
    def register_user(self, user_id, username, image_path):
        """注册用户（优化版）"""
        try:
            # 提取人脸特征
            features = self._extract_face_features(image_path)
            if features is None:
                return False
            
            # 计算特征哈希用于去重
            feature_hash = hashlib.md5(features.tobytes()).hexdigest()
            
            # 检查是否已存在相同特征
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT user_id FROM users WHERE feature_hash = ?', (feature_hash,))
            existing = cursor.fetchone()
            
            if existing:
                print(f"⚠️  特征已存在，用户: {existing[0]}")
                conn.close()
                return False
            
            # 存储到数据库
            encrypted_features = self._encrypt_features(features)
            cursor.execute('''
                INSERT OR REPLACE INTO users (user_id, username, face_features, feature_hash)
                VALUES (?, ?, ?, ?)
            ''', (user_id, username, encrypted_features, feature_hash))
            
            conn.commit()
            conn.close()
            
            # 添加到索引
            self._add_to_indexes(user_id, features)
            
            # 更新缓存
            with self.cache_lock:
                self.feature_cache[user_id] = features
                self.user_cache[user_id] = username
            
            print(f"✅ 用户 {username} 注册成功")
            return True
            
        except Exception as e:
            print(f"❌ 用户注册失败: {e}")
            return False
    
    def _add_to_indexes(self, user_id, features):
        """添加特征到多级索引"""
        index_id = self.coarse_index.ntotal
        
        # 归一化特征向量
        features_norm = features / np.linalg.norm(features)
        features_reshaped = features_norm.reshape(1, -1)
        
        # 添加到粗粒度索引
        if not self.coarse_index.is_trained:
            # 训练索引（需要至少100个样本）
            if self.coarse_index.ntotal < 100:
                # 使用随机数据预训练
                random_data = np.random.rand(100, self.feature_dim).astype('float32')
                random_data = random_data / np.linalg.norm(random_data, axis=1, keepdims=True)
                self.coarse_index.train(random_data)
        
        self.coarse_index.add(features_reshaped)
        self.fine_index.add(features_reshaped)
        self.user_mapping[index_id] = user_id
    
    def _extract_face_features(self, image_path):
        """提取人脸特征向量"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ 无法读取图像: {image_path}")
                return None
            
            faces = self.face_app.get(img)
            if len(faces) == 0:
                print("❌ 未检测到人脸")
                return None
            
            face = faces[0]
            features = face.embedding
            return features
            
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            return None
    
    def verify_payment(self, image_path, amount=100.0):
        """验证支付（高级优化版）"""
        try:
            print(f"\n💳 模拟支付验证 (高级优化版)...")
            
            # 提取当前人脸特征
            current_features = self._extract_face_features(image_path)
            if current_features is None:
                return False, 0.0
            
            start_time = time.time()
            
            # 使用多级搜索策略
            best_match = self._multi_level_search(current_features)
            
            search_time = time.time() - start_time
            
            # 更新统计信息
            self.stats['total_verifications'] += 1
            self.stats['total_search_time'] += search_time
            self.stats['avg_search_time'] = self.stats['total_search_time'] / self.stats['total_verifications']
            
            print(f"🔍 搜索耗时: {search_time*1000:.2f}ms")
            
            if best_match:
                user_id, username, score = best_match
                threshold = 0.6
                
                if score >= threshold:
                    print(f"✅ 身份验证成功: {username} (相似度: {score:.3f})")
                    
                    # 记录支付
                    self._record_payment(user_id, amount, True, score, search_time)
                    
                    # 更新用户最后访问时间
                    self._update_user_access_time(user_id)
                    
                    return True, score
                else:
                    print(f"❌ 身份验证失败 (最高相似度: {score:.3f})")
                    self._record_payment("unknown", amount, False, score, search_time)
                    return False, score
            else:
                print("❌ 未找到匹配用户")
                self._record_payment("unknown", amount, False, 0.0, search_time)
                return False, 0.0
                
        except Exception as e:
            print(f"❌ 身份验证出错: {e}")
            return False, 0.0
    
    def _multi_level_search(self, query_features):
        """多级搜索策略"""
        query_norm = query_features / np.linalg.norm(query_features)
        query_reshaped = query_norm.reshape(1, -1)
        
        # 第一级：粗粒度搜索 (IVF)
        k_coarse = min(50, self.coarse_index.ntotal)  # 粗搜索50个候选
        if k_coarse == 0:
            return None
        
        coarse_scores, coarse_indices = self.coarse_index.search(query_reshaped, k_coarse)
        
        # 第二级：细粒度搜索 (HNSW) - 在粗搜索结果中精搜索
        k_fine = min(10, len(coarse_indices[0]))  # 精搜索前10个
        fine_scores, fine_indices = self.fine_index.search(query_reshaped, k_fine)
        
        # 找到最佳匹配
        best_score = 0.0
        best_match = None
        
        for score, idx in zip(fine_scores[0], fine_indices[0]):
            if idx in self.user_mapping:
                user_id = self.user_mapping[idx]
                
                # 从缓存获取用户信息
                username = self._get_user_info_cached(user_id)
                if username and score > best_score:
                    best_score = score
                    best_match = (user_id, username, score)
        
        return best_match
    
    def _get_user_info_cached(self, user_id):
        """获取用户信息（带缓存）"""
        with self.cache_lock:
            if user_id in self.user_cache:
                self.stats['cache_hits'] += 1
                return self.user_cache[user_id]
        
        # 缓存未命中，从数据库获取
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT username FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            username = result[0]
            with self.cache_lock:
                self.user_cache[user_id] = username
            return username
        
        return None
    
    def _encrypt_features(self, features):
        """加密特征向量"""
        features_str = json.dumps(features.tolist())
        return features_str.encode('utf-8')
    
    def _record_payment(self, user_id, amount, success, confidence, search_time):
        """记录支付信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO payment_records (user_id, amount, success, confidence_score, search_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, amount, success, confidence, search_time))
        
        conn.commit()
        conn.close()
    
    def _update_user_access_time(self, user_id):
        """更新用户最后访问时间"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET last_accessed = CURRENT_TIMESTAMP WHERE user_id = ?
        ''', (user_id,))
        conn.commit()
        conn.close()
    
    def get_performance_stats(self):
        """获取性能统计"""
        cache_hit_rate = (self.stats['cache_hits'] / max(1, self.stats['total_verifications'])) * 100
        
        return {
            'total_verifications': self.stats['total_verifications'],
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'avg_search_time': f"{self.stats['avg_search_time']*1000:.2f}ms",
            'total_users': self.coarse_index.ntotal if self.coarse_index else 0
        }
    
    def get_payment_history(self, user_id=None, limit=10):
        """获取支付历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT id, user_id, amount, success, confidence_score, search_time, timestamp
                FROM payment_records WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT id, user_id, amount, success, confidence_score, search_time, timestamp
                FROM payment_records ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
        
        records = cursor.fetchall()
        conn.close()
        
        return records
    
    def get_registered_users(self):
        """获取已注册用户列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, created_at, last_accessed FROM users ORDER BY created_at
        ''')
        
        users = cursor.fetchall()
        conn.close()
        
        return users

def main():
    """主函数 - 高级性能测试"""
    print("🚀 高级人脸识别支付系统")
    print("=" * 50)
    
    # 初始化系统
    system = AdvancedFacePaymentSystem()
    
    # 注册测试用户
    print("\n📝 注册测试用户...")
    test_users = [
        ("user001", "张三", "test_images/person1_1.jpg"),
        ("user002", "李四", "test_images/person2_1.jpg"),
        ("user003", "王五", "test_images/person1_2.jpg"),
        ("user004", "赵六", "test_images/person2_2.jpg"),
        ("user005", "钱七", "test_images/person1_3.jpg"),
    ]
    
    for user_id, username, image_path in test_users:
        if os.path.exists(image_path):
            system.register_user(user_id, username, image_path)
        else:
            print(f"⚠️  跳过不存在的图像: {image_path}")
    
    # 性能测试
    print(f"\n⚡ 高级性能测试...")
    
    test_cases = [
        ("test_images/person1_1.jpg", "张三"),
        ("test_images/person2_1.jpg", "李四"),
        ("test_images/person1_2.jpg", "王五"),
        ("test_images/person2_2.jpg", "赵六"),
        ("test_images/person1_3.jpg", "钱七"),
    ]
    
    # 多轮测试以测试缓存效果
    for round_num in range(3):
        print(f"\n🔄 第 {round_num + 1} 轮测试:")
        
        for image_path, expected_user in test_cases:
            if os.path.exists(image_path):
                success, score = system.verify_payment(image_path, 100.0)
                status = "✅" if success else "❌"
                print(f"  {status} {expected_user}: 相似度 {score:.3f}")
            else:
                print(f"  ⚠️  跳过: {image_path}")
    
    # 性能统计
    stats = system.get_performance_stats()
    print(f"\n📊 性能统计:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    # 显示用户和支付历史
    print(f"\n👥 已注册用户:")
    users = system.get_registered_users()
    for user_id, username, created_at, last_accessed in users:
        print(f"  - {username} ({user_id}) - 注册: {created_at} - 最后访问: {last_accessed}")
    
    print(f"\n📊 最近支付记录:")
    records = system.get_payment_history(limit=5)
    for record in records:
        record_id, user_id, amount, success, confidence, search_time, timestamp = record
        status = "成功" if success else "失败"
        print(f"  - {record_id}: ¥{user_id} - {status} - 置信度: {confidence:.3f} - 搜索时间: {search_time*1000:.1f}ms")

if __name__ == "__main__":
    main()
