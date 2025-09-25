"""
人脸识别聚类服务配置文件
"""

import os

class Config:
    # 服务配置
    HOST = "0.0.0.0"
    PORT = 8000
    DEBUG = True
    
    # 数据库配置
    DATABASE_PATH = "face_clustering.db"
    
    # 文件上传配置
    UPLOAD_DIR = "uploads"
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # 人脸检测配置
    FACE_DETECTION_CONFIDENCE = 0.5
    FACE_DETECTION_SIZE = (640, 640)
    
    # 聚类配置
    DEFAULT_EPS = 0.4
    DEFAULT_MIN_SAMPLES = 2
    SIMILARITY_THRESHOLD = 0.6
    
    # InsightFace配置
    INSIGHTFACE_MODEL_NAME = "buffalo_l"  # 可选: buffalo_l, buffalo_m, buffalo_s
    INSIGHTFACE_CTX_ID = 0  # GPU设备ID，-1表示CPU
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "WARNING"

# 根据环境变量选择配置
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """获取当前配置"""
    env = os.getenv('FLASK_ENV', 'default')
    return config.get(env, config['default'])
