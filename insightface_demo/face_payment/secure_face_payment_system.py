#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高安全级别人脸识别支付系统
确保不会匹配错人的多重安全机制
"""

import os
import cv2
import json
import sqlite3
import numpy as np
import hashlib
import hmac
import time
import random
from datetime import datetime, timedelta
from insightface import app
import faiss
import pickle
import threading
from collections import defaultdict, deque
import logging
from cryptography.fernet import Fernet
import base64

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecureFacePaymentSystem:
    def __init__(self, db_path="secure_face_payment.db"):
        self.db_path = db_path
        self.face_app = app.FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # 安全配置
        self.security_config = {
            'min_confidence': 0.75,  # 最低置信度阈值
            'max_confidence': 0.95,  # 最高置信度阈值（防止过拟合）
            'liveness_threshold': 0.8,  # 活体检测阈值
            'multi_angle_threshold': 0.7,  # 多角度验证阈值
            'time_window': 30,  # 时间窗口（秒）
            'max_attempts': 3,  # 最大尝试次数
            'cooldown_time': 300,  # 冷却时间（秒）
            'encryption_key': Fernet.generate_key()  # 生成加密密钥
        }
        
        # 特征向量索引
        self.feature_index = None
        self.user_mapping = {}
        self.feature_dim = 512
        
        # 安全状态跟踪
        self.attempt_tracker = defaultdict(list)  # 尝试次数跟踪
        self.user_sessions = {}  # 用户会话
        self.risk_scores = defaultdict(float)  # 风险评分
        self.encryption = Fernet(self.security_config['encryption_key'])
        
        # 活体检测模型（简化版）
        self.liveness_detector = self._init_liveness_detector()
        
        # 初始化
        self._init_database()
        self._load_or_create_index()
        
        logger.info("🔐 高安全级别人脸支付系统初始化完成")
    
    def _init_liveness_detector(self):
        """初始化活体检测器"""
        # 这里使用简化的活体检测，实际应用中需要更复杂的模型
        return {
            'blink_detector': True,
            'motion_detector': True,
            'depth_detector': True
        }
    
    def _init_database(self):
        """初始化安全数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 用户表（增强安全字段）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                face_features BLOB,
                feature_hash TEXT UNIQUE,
                liveness_features BLOB,
                multi_angle_features BLOB,
                risk_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # 支付记录表（增强安全字段）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS payment_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                amount REAL,
                success BOOLEAN,
                confidence_score REAL,
                liveness_score REAL,
                multi_angle_score REAL,
                risk_score REAL,
                device_fingerprint TEXT,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            )
        ''')
        
        # 安全事件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                user_id TEXT,
                severity TEXT,
                description TEXT,
                metadata TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建安全索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_hash ON users(feature_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_active ON users(is_active)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_payment_timestamp ON payment_records(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_security_events ON security_events(timestamp)')
        
        conn.commit()
        conn.close()
        logger.info("✅ 安全数据库初始化完成")
    
    def _load_or_create_index(self):
        """加载或创建FAISS索引"""
        index_file = "secure_face_features.index"
        mapping_file = "secure_user_mapping.pkl"
        
        if os.path.exists(index_file) and os.path.exists(mapping_file):
            self.feature_index = faiss.read_index(index_file)
            with open(mapping_file, 'rb') as f:
                self.user_mapping = pickle.load(f)
            logger.info(f"✅ 加载安全特征索引，包含 {self.feature_index.ntotal} 个用户")
        else:
            self.feature_index = faiss.IndexFlatIP(self.feature_dim)
            self.user_mapping = {}
            logger.info("✅ 创建新的安全特征索引")
    
    def _save_index(self):
        """保存索引到文件"""
        if self.feature_index and self.user_mapping:
            faiss.write_index(self.feature_index, "secure_face_features.index")
            with open("secure_user_mapping.pkl", 'wb') as f:
                pickle.dump(self.user_mapping, f)
            logger.info("✅ 安全特征索引已保存")
    
    def register_user(self, user_id, username, image_paths, liveness_video_path=None):
        """注册用户（多角度+活体检测）"""
        try:
            logger.info(f"📝 开始注册用户: {username}")
            
            # 1. 多角度人脸特征提取
            multi_angle_features = []
            for i, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    logger.warning(f"⚠️  图像文件不存在: {image_path}")
                    continue
                
                features = self._extract_face_features(image_path)
                if features is not None:
                    multi_angle_features.append(features)
                    logger.info(f"✅ 角度 {i+1} 特征提取成功")
                else:
                    logger.warning(f"❌ 角度 {i+1} 特征提取失败")
            
            if len(multi_angle_features) < 2:
                logger.error("❌ 至少需要2个角度的清晰人脸图像")
                return False
            
            # 2. 计算平均特征向量
            avg_features = np.mean(multi_angle_features, axis=0)
            
            # 3. 活体检测特征提取
            liveness_features = None
            if liveness_video_path and os.path.exists(liveness_video_path):
                liveness_features = self._extract_liveness_features(liveness_video_path)
            
            # 4. 特征去重检查
            feature_hash = hashlib.sha256(avg_features.tobytes()).hexdigest()
            if self._check_duplicate_features(feature_hash):
                logger.error("❌ 检测到重复的人脸特征，可能存在身份冒用")
                self._log_security_event("DUPLICATE_FEATURES", user_id, "HIGH", 
                                       f"用户 {username} 注册时检测到重复特征")
                return False
            
            # 5. 存储到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 加密存储
            encrypted_features = self._encrypt_data(avg_features.tobytes())
            encrypted_multi_angle = self._encrypt_data(pickle.dumps(multi_angle_features))
            encrypted_liveness = self._encrypt_data(pickle.dumps(liveness_features)) if liveness_features else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, username, face_features, feature_hash, liveness_features, multi_angle_features)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, username, encrypted_features, feature_hash, encrypted_liveness, encrypted_multi_angle))
            
            conn.commit()
            conn.close()
            
            # 6. 添加到索引
            self._add_to_index(user_id, avg_features)
            
            logger.info(f"✅ 用户 {username} 注册成功，包含 {len(multi_angle_features)} 个角度")
            return True
            
        except Exception as e:
            logger.error(f"❌ 用户注册失败: {e}")
            self._log_security_event("REGISTRATION_FAILED", user_id, "MEDIUM", str(e))
            return False
    
    def verify_payment(self, image_path, amount=100.0, device_fingerprint=None, ip_address=None):
        """验证支付（多重安全验证）"""
        try:
            logger.info(f"💳 开始支付验证，金额: ¥{amount}")
            
            # 1. 基础安全检查
            if not self._basic_security_check(device_fingerprint, ip_address):
                return False, 0.0, "安全检查失败"
            
            # 2. 提取当前人脸特征
            current_features = self._extract_face_features(image_path)
            if current_features is None:
                self._log_security_event("FACE_DETECTION_FAILED", "unknown", "LOW", "未检测到人脸")
                return False, 0.0, "未检测到人脸"
            
            # 3. 活体检测
            liveness_score = self._detect_liveness(image_path)
            if liveness_score < self.security_config['liveness_threshold']:
                logger.warning(f"❌ 活体检测失败，得分: {liveness_score:.3f}")
                self._log_security_event("LIVENESS_FAILED", "unknown", "HIGH", f"活体检测失败，得分: {liveness_score}")
                return False, 0.0, "活体检测失败"
            
            # 4. 多级特征匹配
            start_time = time.time()
            match_result = self._multi_level_matching(current_features)
            search_time = time.time() - start_time
            
            if not match_result:
                logger.warning("❌ 未找到匹配用户")
                return False, 0.0, "未找到匹配用户"
            
            user_id, username, confidence, multi_angle_score = match_result
            
            # 5. 综合安全评分
            risk_score = self._calculate_risk_score(user_id, confidence, liveness_score, multi_angle_score)
            
            # 6. 最终验证决策
            if self._final_verification_decision(confidence, liveness_score, multi_angle_score, risk_score):
                logger.info(f"✅ 身份验证成功: {username}")
                logger.info(f"   - 人脸相似度: {confidence:.3f}")
                logger.info(f"   - 活体检测: {liveness_score:.3f}")
                logger.info(f"   - 多角度验证: {multi_angle_score:.3f}")
                logger.info(f"   - 风险评分: {risk_score:.3f}")
                logger.info(f"   - 搜索耗时: {search_time*1000:.2f}ms")
                
                # 记录成功支付
                self._record_payment(user_id, amount, True, confidence, liveness_score, 
                                   multi_angle_score, risk_score, device_fingerprint, ip_address)
                
                # 更新用户状态
                self._update_user_access(user_id)
                
                return True, confidence, "验证成功"
            else:
                logger.warning(f"❌ 身份验证失败")
                logger.warning(f"   - 人脸相似度: {confidence:.3f}")
                logger.warning(f"   - 活体检测: {liveness_score:.3f}")
                logger.warning(f"   - 多角度验证: {multi_angle_score:.3f}")
                logger.warning(f"   - 风险评分: {risk_score:.3f}")
                
                # 记录失败支付
                self._record_payment("unknown", amount, False, confidence, liveness_score, 
                                   multi_angle_score, risk_score, device_fingerprint, ip_address)
                
                return False, confidence, "验证失败"
                
        except Exception as e:
            logger.error(f"❌ 支付验证出错: {e}")
            self._log_security_event("VERIFICATION_ERROR", "unknown", "HIGH", str(e))
            return False, 0.0, "系统错误"
    
    def _basic_security_check(self, device_fingerprint, ip_address):
        """基础安全检查"""
        # 检查尝试次数限制
        if device_fingerprint:
            attempts = self.attempt_tracker[device_fingerprint]
            current_time = time.time()
            # 清理过期记录
            attempts[:] = [t for t in attempts if current_time - t < self.security_config['cooldown_time']]
            
            if len(attempts) >= self.security_config['max_attempts']:
                logger.warning(f"❌ 设备 {device_fingerprint} 尝试次数过多")
                self._log_security_event("TOO_MANY_ATTEMPTS", "unknown", "HIGH", 
                                       f"设备 {device_fingerprint} 尝试次数过多")
                return False
            
            # 记录当前尝试
            attempts.append(current_time)
        
        return True
    
    def _multi_level_matching(self, query_features):
        """多级特征匹配"""
        if self.feature_index.ntotal == 0:
            return None
        
        # 第一级：快速粗搜索
        k = min(10, self.feature_index.ntotal)
        query_norm = query_features / np.linalg.norm(query_features)
        scores, indices = self.feature_index.search(query_norm.reshape(1, -1), k)
        
        best_score = 0.0
        best_match = None
        
        # 第二级：精确匹配和多角度验证
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.user_mapping:
                user_id = self.user_mapping[idx]
                
                # 获取用户多角度特征进行验证
                multi_angle_score = self._verify_multi_angle(user_id, query_features)
                
                # 综合评分
                combined_score = (score * 0.6 + multi_angle_score * 0.4)
                
                if combined_score > best_score:
                    best_score = combined_score
                    username = self._get_username(user_id)
                    best_match = (user_id, username, score, multi_angle_score)
        
        return best_match
    
    def _verify_multi_angle(self, user_id, query_features):
        """多角度验证"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT multi_angle_features FROM users WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if not result or not result[0]:
                return 0.0
            
            # 解密多角度特征
            encrypted_features = result[0]
            multi_angle_features = pickle.loads(self._decrypt_data(encrypted_features))
            
            # 与每个角度比较
            max_similarity = 0.0
            for angle_features in multi_angle_features:
                similarity = np.dot(query_features, angle_features) / (
                    np.linalg.norm(query_features) * np.linalg.norm(angle_features)
                )
                max_similarity = max(max_similarity, similarity)
            
            return max_similarity
            
        except Exception as e:
            logger.error(f"多角度验证失败: {e}")
            return 0.0
    
    def _detect_liveness(self, image_path):
        """活体检测（简化版）"""
        try:
            # 这里使用简化的活体检测，实际应用中需要更复杂的模型
            # 1. 检测图像质量
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            
            # 2. 检测人脸清晰度
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            clarity_score = min(laplacian_var / 1000.0, 1.0)
            
            # 3. 检测图像真实性（简化版）
            # 实际应用中需要检测照片、视频、3D面具等攻击
            authenticity_score = 0.9  # 简化版假设为真实
            
            # 4. 综合活体评分
            liveness_score = (clarity_score * 0.6 + authenticity_score * 0.4)
            
            return liveness_score
            
        except Exception as e:
            logger.error(f"活体检测失败: {e}")
            return 0.0
    
    def _calculate_risk_score(self, user_id, confidence, liveness_score, multi_angle_score):
        """计算风险评分"""
        base_risk = 0.0
        
        # 1. 置信度风险
        if confidence < 0.8:
            base_risk += 0.3
        elif confidence < 0.9:
            base_risk += 0.1
        
        # 2. 活体检测风险
        if liveness_score < 0.9:
            base_risk += 0.2
        
        # 3. 多角度验证风险
        if multi_angle_score < 0.8:
            base_risk += 0.2
        
        # 4. 用户历史风险
        user_risk = self.risk_scores.get(user_id, 0.0)
        base_risk += user_risk * 0.3
        
        # 5. 时间风险（深夜等异常时间）
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 23:
            base_risk += 0.1
        
        return min(base_risk, 1.0)
    
    def _final_verification_decision(self, confidence, liveness_score, multi_angle_score, risk_score):
        """最终验证决策"""
        # 1. 基础阈值检查
        if confidence < self.security_config['min_confidence']:
            return False
        
        if liveness_score < self.security_config['liveness_threshold']:
            return False
        
        if multi_angle_score < self.security_config['multi_angle_threshold']:
            return False
        
        # 2. 风险评分检查
        if risk_score > 0.5:
            return False
        
        # 3. 综合评分
        overall_score = (
            confidence * 0.4 +
            liveness_score * 0.3 +
            multi_angle_score * 0.2 +
            (1 - risk_score) * 0.1
        )
        
        return overall_score >= 0.8
    
    def _extract_face_features(self, image_path):
        """提取人脸特征向量"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            faces = self.face_app.get(img)
            if len(faces) == 0:
                return None
            
            face = faces[0]
            return face.embedding
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None
    
    def _extract_liveness_features(self, video_path):
        """提取活体检测特征"""
        # 简化版活体特征提取
        # 实际应用中需要分析视频中的眨眼、张嘴等动作
        return np.random.rand(128)  # 模拟活体特征
    
    def _check_duplicate_features(self, feature_hash):
        """检查重复特征"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT user_id FROM users WHERE feature_hash = ?', (feature_hash,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    
    def _add_to_index(self, user_id, features):
        """添加特征到索引"""
        index_id = self.feature_index.ntotal
        self.feature_index.add(features.reshape(1, -1))
        self.user_mapping[index_id] = user_id
        self._save_index()
    
    def _get_username(self, user_id):
        """获取用户名"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT username FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "Unknown"
    
    def _encrypt_data(self, data):
        """加密数据"""
        return self.encryption.encrypt(data)
    
    def _decrypt_data(self, encrypted_data):
        """解密数据"""
        return self.encryption.decrypt(encrypted_data)
    
    def _record_payment(self, user_id, amount, success, confidence, liveness_score, 
                       multi_angle_score, risk_score, device_fingerprint, ip_address):
        """记录支付信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        session_id = hashlib.md5(f"{user_id}_{time.time()}".encode()).hexdigest()
        
        cursor.execute('''
            INSERT INTO payment_records 
            (user_id, amount, success, confidence_score, liveness_score, multi_angle_score, 
             risk_score, device_fingerprint, ip_address, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, amount, success, confidence, liveness_score, multi_angle_score, 
              risk_score, device_fingerprint, ip_address, session_id))
        
        conn.commit()
        conn.close()
    
    def _update_user_access(self, user_id):
        """更新用户访问信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users 
            SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
            WHERE user_id = ?
        ''', (user_id,))
        conn.commit()
        conn.close()
    
    def _log_security_event(self, event_type, user_id, severity, description, metadata=None):
        """记录安全事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO security_events (event_type, user_id, severity, description, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (event_type, user_id, severity, description, metadata_json))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"🚨 安全事件: {event_type} - {description}")
    
    def get_security_report(self):
        """获取安全报告"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 统计信息
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = TRUE')
        active_users = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM payment_records WHERE success = TRUE')
        successful_payments = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM payment_records WHERE success = FALSE')
        failed_payments = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM security_events WHERE timestamp > datetime("now", "-24 hours")')
        recent_events = cursor.fetchone()[0]
        
        # 最近的安全事件
        cursor.execute('''
            SELECT event_type, severity, description, timestamp 
            FROM security_events 
            ORDER BY timestamp DESC LIMIT 10
        ''')
        recent_security_events = cursor.fetchall()
        
        conn.close()
        
        return {
            'active_users': active_users,
            'successful_payments': successful_payments,
            'failed_payments': failed_payments,
            'recent_events': recent_events,
            'security_events': recent_security_events
        }

def main():
    """主函数 - 高安全级别测试"""
    print("🔐 高安全级别人脸识别支付系统")
    print("=" * 60)
    
    # 初始化系统
    system = SecureFacePaymentSystem()
    
    # 注册测试用户（多角度）
    print("\n📝 注册测试用户（多角度验证）...")
    test_users = [
        ("user001", "张三", ["test_images/person1_1.jpg", "test_images/person1_2.jpg"]),
        ("user002", "李四", ["test_images/person2_1.jpg", "test_images/person2_2.jpg"]),
    ]
    
    for user_id, username, image_paths in test_users:
        # 检查图像文件是否存在
        existing_paths = [path for path in image_paths if os.path.exists(path)]
        if len(existing_paths) >= 2:
            system.register_user(user_id, username, existing_paths)
        else:
            print(f"⚠️  跳过用户 {username}，图像文件不足")
    
    # 安全测试
    print(f"\n🔒 安全测试...")
    
    test_cases = [
        ("test_images/person1_1.jpg", "张三"),
        ("test_images/person2_1.jpg", "李四"),
    ]
    
    for image_path, expected_user in test_cases:
        if os.path.exists(image_path):
            print(f"\n🧪 测试用户: {expected_user}")
            success, confidence, message = system.verify_payment(
                image_path, 100.0, 
                device_fingerprint=f"device_{random.randint(1000, 9999)}",
                ip_address=f"192.168.1.{random.randint(1, 254)}"
            )
            
            status = "✅" if success else "❌"
            print(f"  {status} 结果: {message}")
            print(f"  📊 置信度: {confidence:.3f}")
        else:
            print(f"⚠️  跳过不存在的图像: {image_path}")
    
    # 安全报告
    print(f"\n📊 安全报告:")
    report = system.get_security_report()
    print(f"  - 活跃用户: {report['active_users']}")
    print(f"  - 成功支付: {report['successful_payments']}")
    print(f"  - 失败支付: {report['failed_payments']}")
    print(f"  - 24小时内安全事件: {report['recent_events']}")
    
    if report['security_events']:
        print(f"\n🚨 最近安全事件:")
        for event_type, severity, description, timestamp in report['security_events']:
            print(f"  - [{severity}] {event_type}: {description} ({timestamp})")

if __name__ == "__main__":
    main()
