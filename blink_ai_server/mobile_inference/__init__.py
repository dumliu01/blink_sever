"""
移动端推理模块
提供量化模型的移动端推理功能
"""

from .onnx_inference import ONNXInference
from .mobile_face_service import MobileFaceService

__version__ = "1.0.0"
__all__ = [
    "ONNXInference",
    "MobileFaceService"
]
