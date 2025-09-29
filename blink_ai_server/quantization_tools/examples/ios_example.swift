//
//  iOSExample.swift
//  InsightFace Mobile Inference Example
//
//  Created by AI Assistant
//  Copyright © 2024. All rights reserved.
//

import UIKit
import InsightFaceInference

class ViewController: UIViewController {
    
    // MARK: - Properties
    
    private var insightFace: InsightFaceInference?
    private let imagePicker = UIImagePickerController()
    
    // MARK: - UI Elements
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!
    @IBOutlet weak var processButton: UIButton!
    
    // MARK: - Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupInsightFace()
    }
    
    // MARK: - Setup
    
    private func setupUI() {
        title = "InsightFace 移动端推理"
        
        // 配置图像选择器
        imagePicker.delegate = self
        imagePicker.sourceType = .photoLibrary
        imagePicker.allowsEditing = false
        
        // 配置按钮
        processButton.setTitle("选择图像并处理", for: .normal)
        processButton.backgroundColor = .systemBlue
        processButton.setTitleColor(.white, for: .normal)
        processButton.layer.cornerRadius = 8
        
        // 配置结果标签
        resultLabel.text = "请选择图像开始处理"
        resultLabel.numberOfLines = 0
        resultLabel.textAlignment = .center
    }
    
    private func setupInsightFace() {
        do {
            // 初始化 InsightFace 推理器
            // 这里需要将量化后的模型文件添加到项目中
            let modelPath = Bundle.main.path(forResource: "buffalo_l_int8", ofType: "onnx") ?? ""
            
            insightFace = try InsightFaceInference(
                modelPath: modelPath,
                modelType: .onnx,
                inputName: "input",
                outputNames: ["output"]
            )
            
            print("InsightFace 初始化成功")
        } catch {
            print("InsightFace 初始化失败: \(error.localizedDescription)")
            showAlert(title: "错误", message: "模型初始化失败: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Actions
    
    @IBAction func processButtonTapped(_ sender: UIButton) {
        present(imagePicker, animated: true)
    }
    
    // MARK: - Processing
    
    private func processImage(_ image: UIImage) {
        guard let insightFace = insightFace else {
            showAlert(title: "错误", message: "InsightFace 未初始化")
            return
        }
        
        // 显示加载指示器
        showLoadingIndicator()
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                // 执行人脸检测
                let detections = try insightFace.detectFaces(in: image)
                
                // 执行人脸识别
                let features = try insightFace.recognizeFace(in: image)
                
                // 在主线程更新UI
                DispatchQueue.main.async {
                    self?.hideLoadingIndicator()
                    self?.updateUI(with: detections, features: features)
                }
                
            } catch {
                DispatchQueue.main.async {
                    self?.hideLoadingIndicator()
                    self?.showAlert(title: "处理失败", message: error.localizedDescription)
                }
            }
        }
    }
    
    private func updateUI(with detections: [FaceDetection], features: [Float]) {
        // 更新图像显示
        imageView.image = drawDetections(on: imageView.image ?? UIImage(), detections: detections)
        
        // 更新结果标签
        var resultText = "检测到 \(detections.count) 个人脸\n\n"
        
        for (index, detection) in detections.enumerated() {
            resultText += "人脸 \(index + 1):\n"
            resultText += "  置信度: \(String(format: "%.2f", detection.confidence))\n"
            resultText += "  位置: \(detection.boundingBox)\n"
        }
        
        if !features.isEmpty {
            resultText += "\n特征向量维度: \(features.count)"
        }
        
        resultLabel.text = resultText
    }
    
    private func drawDetections(on image: UIImage, detections: [FaceDetection]) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        defer { UIGraphicsEndImageContext() }
        
        // 绘制原图
        image.draw(in: CGRect(origin: .zero, size: image.size))
        
        // 绘制检测框
        let context = UIGraphicsGetCurrentContext()
        context?.setStrokeColor(UIColor.red.cgColor)
        context?.setLineWidth(2.0)
        
        for detection in detections {
            let rect = detection.boundingBox
            context?.stroke(rect)
            
            // 绘制置信度
            let confidenceText = String(format: "%.2f", detection.confidence)
            let attributes: [NSAttributedString.Key: Any] = [
                .foregroundColor: UIColor.red,
                .font: UIFont.systemFont(ofSize: 16)
            ]
            confidenceText.draw(at: CGPoint(x: rect.minX, y: rect.minY - 20), withAttributes: attributes)
        }
        
        return UIGraphicsGetImageFromCurrentImageContext() ?? image
    }
    
    // MARK: - Helper Methods
    
    private func showLoadingIndicator() {
        let activityIndicator = UIActivityIndicatorView(style: .large)
        activityIndicator.tag = 999
        activityIndicator.center = view.center
        activityIndicator.startAnimating()
        view.addSubview(activityIndicator)
    }
    
    private func hideLoadingIndicator() {
        view.subviews.first { $0.tag == 999 }?.removeFromSuperview()
    }
    
    private func showAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "确定", style: .default))
        present(alert, animated: true)
    }
}

// MARK: - UIImagePickerControllerDelegate

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true)
        
        if let image = info[.originalImage] as? UIImage {
            imageView.image = image
            processImage(image)
        }
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true)
    }
}

// MARK: - Batch Processing Example

extension ViewController {
    
    /// 批量处理示例
    func batchProcessExample() {
        guard let insightFace = insightFace else { return }
        
        // 假设有一组图像
        let images: [UIImage] = [] // 从相册或其他来源获取图像
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                let results = try insightFace.batchProcess(images: images)
                
                DispatchQueue.main.async {
                    self?.handleBatchResults(results)
                }
                
            } catch {
                DispatchQueue.main.async {
                    self?.showAlert(title: "批量处理失败", message: error.localizedDescription)
                }
            }
        }
    }
    
    private func handleBatchResults(_ results: [ProcessingResult]) {
        var successCount = 0
        var errorCount = 0
        
        for result in results {
            if result.error == nil {
                successCount += 1
                print("处理成功: 检测到 \(result.detections.count) 个人脸")
            } else {
                errorCount += 1
                print("处理失败: \(result.error?.localizedDescription ?? "未知错误")")
            }
        }
        
        showAlert(title: "批量处理完成", 
                 message: "成功: \(successCount), 失败: \(errorCount)")
    }
}

// MARK: - Performance Testing

extension ViewController {
    
    /// 性能测试示例
    func performanceTestExample() {
        guard let insightFace = insightFace,
              let testImage = imageView.image else { return }
        
        let iterations = 100
        var times: [TimeInterval] = []
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            for i in 0..<iterations {
                let startTime = CFAbsoluteTimeGetCurrent()
                
                do {
                    _ = try insightFace.detectFaces(in: testImage)
                } catch {
                    print("性能测试失败: \(error.localizedDescription)")
                    return
                }
                
                let endTime = CFAbsoluteTimeGetCurrent()
                times.append(endTime - startTime)
                
                if i % 10 == 0 {
                    print("性能测试进度: \(i)/\(iterations)")
                }
            }
            
            let averageTime = times.reduce(0, +) / Double(times.count)
            let fps = 1.0 / averageTime
            
            DispatchQueue.main.async {
                self?.showAlert(title: "性能测试结果", 
                               message: "平均推理时间: \(String(format: "%.4f", averageTime))s\n推理速度: \(String(format: "%.2f", fps)) FPS")
            }
        }
    }
}
