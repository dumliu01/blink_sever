"""
InsightFace 演示测试脚本
用于测试各个功能模块的基本功能
"""

import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Any
import json
import time

# 导入各个功能模块
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from face_clustering import FaceClusterer
from face_attributes import FaceAttributeAnalyzer
from face_quality import FaceQualityAssessor
from face_liveness import FaceLivenessDetector


class InsightFaceTester:
    """InsightFace功能测试器"""
    
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        初始化测试器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.test_results = {}
        self.test_images = []
        
        print("=== InsightFace 功能测试器 ===")
        print(f"使用模型: {model_name}")
    
    def setup_test_images(self, image_dir: str = "test_images"):
        """
        设置测试图像
        
        Args:
            image_dir: 测试图像目录
        """
        if not os.path.exists(image_dir):
            print(f"测试图像目录不存在: {image_dir}")
            return False
        
        # 查找所有图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([f for f in os.listdir(image_dir) 
                              if f.lower().endswith(ext)])
        
        if not image_files:
            print(f"在目录 {image_dir} 中没有找到图像文件")
            return False
        
        self.test_images = [os.path.join(image_dir, f) for f in image_files]
        print(f"找到 {len(self.test_images)} 个测试图像")
        return True
    
    def test_face_detection(self) -> Dict[str, Any]:
        """测试人脸检测功能"""
        print("\n--- 测试人脸检测功能 ---")
        
        try:
            detector = FaceDetector(self.model_name)
            results = {
                'success': True,
                'total_images': len(self.test_images),
                'detected_images': 0,
                'total_faces': 0,
                'avg_confidence': 0.0,
                'processing_time': 0.0,
                'errors': []
            }
            
            start_time = time.time()
            confidences = []
            
            for img_path in self.test_images:
                try:
                    faces = detector.detect_faces(img_path)
                    if faces:
                        results['detected_images'] += 1
                        results['total_faces'] += len(faces)
                        confidences.extend([face['confidence'] for face in faces])
                except Exception as e:
                    results['errors'].append(f"{img_path}: {str(e)}")
            
            results['processing_time'] = time.time() - start_time
            results['avg_confidence'] = np.mean(confidences) if confidences else 0.0
            
            print(f"✓ 人脸检测测试完成")
            print(f"  成功检测图像: {results['detected_images']}/{results['total_images']}")
            print(f"  检测到人脸总数: {results['total_faces']}")
            print(f"  平均置信度: {results['avg_confidence']:.3f}")
            print(f"  处理时间: {results['processing_time']:.2f}秒")
            
            return results
            
        except Exception as e:
            print(f"✗ 人脸检测测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_face_recognition(self) -> Dict[str, Any]:
        """测试人脸识别功能"""
        print("\n--- 测试人脸识别功能 ---")
        
        try:
            recognizer = FaceRecognizer(self.model_name)
            results = {
                'success': True,
                'registration_success': 0,
                'recognition_success': 0,
                'total_images': len(self.test_images),
                'processing_time': 0.0,
                'errors': []
            }
            
            start_time = time.time()
            
            # 注册测试人员
            for i, img_path in enumerate(self.test_images):
                try:
                    person_id = f"person_{i+1}"
                    person_name = f"测试人员{i+1}"
                    if recognizer.register_person(person_id, person_name, img_path):
                        results['registration_success'] += 1
                except Exception as e:
                    results['errors'].append(f"注册 {img_path}: {str(e)}")
            
            # 测试识别
            for img_path in self.test_images:
                try:
                    result = recognizer.identify_face(img_path)
                    if result['identified']:
                        results['recognition_success'] += 1
                except Exception as e:
                    results['errors'].append(f"识别 {img_path}: {str(e)}")
            
            results['processing_time'] = time.time() - start_time
            
            print(f"✓ 人脸识别测试完成")
            print(f"  注册成功: {results['registration_success']}/{results['total_images']}")
            print(f"  识别成功: {results['recognition_success']}/{results['total_images']}")
            print(f"  处理时间: {results['processing_time']:.2f}秒")
            
            return results
            
        except Exception as e:
            print(f"✗ 人脸识别测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_face_clustering(self) -> Dict[str, Any]:
        """测试人脸聚类功能"""
        print("\n--- 测试人脸聚类功能 ---")
        
        try:
            clusterer = FaceClusterer(self.model_name)
            results = {
                'success': True,
                'added_faces': 0,
                'clustering_success': False,
                'total_clusters': 0,
                'processing_time': 0.0,
                'errors': []
            }
            
            start_time = time.time()
            
            # 添加人脸到聚类器
            for img_path in self.test_images:
                try:
                    faces = clusterer._extract_faces_from_image(img_path)
                    for face in faces:
                        clusterer._save_face_embedding(img_path, face)
                        results['added_faces'] += 1
                except Exception as e:
                    results['errors'].append(f"添加 {img_path}: {str(e)}")
            
            # 执行聚类
            try:
                cluster_result = clusterer.cluster_faces_dbscan(eps=0.4, min_samples=2)
                results['clustering_success'] = True
                results['total_clusters'] = cluster_result['total_clusters']
            except Exception as e:
                results['errors'].append(f"聚类: {str(e)}")
            
            results['processing_time'] = time.time() - start_time
            
            print(f"✓ 人脸聚类测试完成")
            print(f"  添加人脸数: {results['added_faces']}")
            print(f"  聚类成功: {results['clustering_success']}")
            print(f"  聚类数量: {results['total_clusters']}")
            print(f"  处理时间: {results['processing_time']:.2f}秒")
            
            return results
            
        except Exception as e:
            print(f"✗ 人脸聚类测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_face_attributes(self) -> Dict[str, Any]:
        """测试人脸属性分析功能"""
        print("\n--- 测试人脸属性分析功能 ---")
        
        try:
            analyzer = FaceAttributeAnalyzer(self.model_name)
            results = {
                'success': True,
                'analyzed_images': 0,
                'total_faces': 0,
                'avg_age': 0.0,
                'gender_distribution': {'男': 0, '女': 0},
                'emotion_distribution': {},
                'processing_time': 0.0,
                'errors': []
            }
            
            start_time = time.time()
            ages = []
            genders = []
            emotions = []
            
            for img_path in self.test_images:
                try:
                    attributes = analyzer.analyze_face_attributes(img_path)
                    if attributes:
                        results['analyzed_images'] += 1
                        results['total_faces'] += len(attributes)
                        
                        for attr in attributes:
                            ages.append(attr['age'])
                            genders.append(attr['gender'])
                            emotions.append(attr['emotion']['emotion'])
                except Exception as e:
                    results['errors'].append(f"{img_path}: {str(e)}")
            
            # 统计结果
            if ages:
                results['avg_age'] = np.mean(ages)
            
            for gender in genders:
                results['gender_distribution'][gender] += 1
            
            for emotion in emotions:
                results['emotion_distribution'][emotion] = results['emotion_distribution'].get(emotion, 0) + 1
            
            results['processing_time'] = time.time() - start_time
            
            print(f"✓ 人脸属性分析测试完成")
            print(f"  分析图像数: {results['analyzed_images']}/{len(self.test_images)}")
            print(f"  检测人脸数: {results['total_faces']}")
            print(f"  平均年龄: {results['avg_age']:.1f}岁")
            print(f"  性别分布: {results['gender_distribution']}")
            print(f"  表情分布: {results['emotion_distribution']}")
            print(f"  处理时间: {results['processing_time']:.2f}秒")
            
            return results
            
        except Exception as e:
            print(f"✗ 人脸属性分析测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_face_quality(self) -> Dict[str, Any]:
        """测试人脸质量评估功能"""
        print("\n--- 测试人脸质量评估功能 ---")
        
        try:
            assessor = FaceQualityAssessor(self.model_name)
            results = {
                'success': True,
                'assessed_images': 0,
                'avg_quality_score': 0.0,
                'quality_distribution': {'优秀': 0, '良好': 0, '一般': 0, '差': 0},
                'common_issues': {},
                'processing_time': 0.0,
                'errors': []
            }
            
            start_time = time.time()
            quality_scores = []
            
            for img_path in self.test_images:
                try:
                    assessment = assessor.assess_quality(img_path)
                    if 'error' not in assessment:
                        results['assessed_images'] += 1
                        quality_scores.append(assessment['quality_score'])
                        results['quality_distribution'][assessment['quality_level']] += 1
                        
                        # 统计常见问题
                        for issue in assessment['issues']:
                            results['common_issues'][issue] = results['common_issues'].get(issue, 0) + 1
                except Exception as e:
                    results['errors'].append(f"{img_path}: {str(e)}")
            
            if quality_scores:
                results['avg_quality_score'] = np.mean(quality_scores)
            
            results['processing_time'] = time.time() - start_time
            
            print(f"✓ 人脸质量评估测试完成")
            print(f"  评估图像数: {results['assessed_images']}/{len(self.test_images)}")
            print(f"  平均质量评分: {results['avg_quality_score']:.3f}")
            print(f"  质量分布: {results['quality_distribution']}")
            print(f"  常见问题: {results['common_issues']}")
            print(f"  处理时间: {results['processing_time']:.2f}秒")
            
            return results
            
        except Exception as e:
            print(f"✗ 人脸质量评估测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_face_liveness(self) -> Dict[str, Any]:
        """测试人脸活体检测功能"""
        print("\n--- 测试人脸活体检测功能 ---")
        
        try:
            detector = FaceLivenessDetector(self.model_name)
            results = {
                'success': True,
                'detected_images': 0,
                'live_count': 0,
                'spoof_count': 0,
                'avg_confidence': 0.0,
                'spoof_types': {},
                'processing_time': 0.0,
                'errors': []
            }
            
            start_time = time.time()
            confidences = []
            
            for img_path in self.test_images:
                try:
                    detection = detector.detect_liveness(img_path)
                    if 'error' not in detection:
                        results['detected_images'] += 1
                        confidences.append(detection['confidence'])
                        
                        if detection['is_live']:
                            results['live_count'] += 1
                        else:
                            results['spoof_count'] += 1
                            spoof_type = detection['spoof_type']
                            results['spoof_types'][spoof_type] = results['spoof_types'].get(spoof_type, 0) + 1
                except Exception as e:
                    results['errors'].append(f"{img_path}: {str(e)}")
            
            if confidences:
                results['avg_confidence'] = np.mean(confidences)
            
            results['processing_time'] = time.time() - start_time
            
            print(f"✓ 人脸活体检测测试完成")
            print(f"  检测图像数: {results['detected_images']}/{len(self.test_images)}")
            print(f"  活体数量: {results['live_count']}")
            print(f"  非活体数量: {results['spoof_count']}")
            print(f"  平均置信度: {results['avg_confidence']:.3f}")
            print(f"  欺骗类型: {results['spoof_types']}")
            print(f"  处理时间: {results['processing_time']:.2f}秒")
            
            return results
            
        except Exception as e:
            print(f"✗ 人脸活体检测测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("\n=== 开始运行所有测试 ===")
        
        # 设置测试图像
        if not self.setup_test_images():
            return {'success': False, 'error': '无法设置测试图像'}
        
        # 运行各项测试
        tests = [
            ('face_detection', self.test_face_detection),
            ('face_recognition', self.test_face_recognition),
            ('face_clustering', self.test_face_clustering),
            ('face_attributes', self.test_face_attributes),
            ('face_quality', self.test_face_quality),
            ('face_liveness', self.test_face_liveness)
        ]
        
        all_results = {}
        total_time = 0
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name.upper()} {'='*20}")
            start_time = time.time()
            
            result = test_func()
            all_results[test_name] = result
            
            test_time = time.time() - start_time
            total_time += test_time
            
            print(f"测试 {test_name} 完成，耗时: {test_time:.2f}秒")
        
        # 生成测试总结
        summary = self._generate_test_summary(all_results, total_time)
        all_results['summary'] = summary
        
        print(f"\n{'='*50}")
        print("所有测试完成！")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"成功测试: {summary['successful_tests']}/{len(tests)}")
        print(f"失败测试: {summary['failed_tests']}/{len(tests)}")
        
        return all_results
    
    def _generate_test_summary(self, results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """生成测试总结"""
        successful_tests = 0
        failed_tests = 0
        
        for test_name, result in results.items():
            if test_name == 'summary':
                continue
            
            if result.get('success', False):
                successful_tests += 1
            else:
                failed_tests += 1
        
        return {
            'total_tests': len(results) - 1,  # 减去summary本身
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'total_time': total_time,
            'success_rate': successful_tests / (successful_tests + failed_tests) if (successful_tests + failed_tests) > 0 else 0
        }
    
    def save_test_results(self, results: Dict[str, Any], output_path: str = "test_results.json"):
        """保存测试结果"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"测试结果已保存到: {output_path}")
        except Exception as e:
            print(f"保存测试结果失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='InsightFace 功能测试')
    parser.add_argument('--model', type=str, default='buffalo_l',
                       choices=['buffalo_l', 'buffalo_m', 'buffalo_s'],
                       help='InsightFace模型名称')
    parser.add_argument('--test-dir', type=str, default='test_images',
                       help='测试图像目录')
    parser.add_argument('--output', type=str, default='test_results.json',
                       help='测试结果输出文件')
    
    args = parser.parse_args()
    
    try:
        # 创建测试器
        tester = InsightFaceTester(args.model)
        
        # 运行所有测试
        results = tester.run_all_tests()
        
        # 保存结果
        tester.save_test_results(results, args.output)
        
    except Exception as e:
        print(f"测试执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
