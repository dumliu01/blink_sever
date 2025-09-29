//
//  InsightFaceInference.swift
//  InsightFace Mobile Inference
//
//  Created by AI Assistant
//  Copyright © 2024. All rights reserved.
//

import Foundation
import CoreML
import Vision
import UIKit
import Accelerate

/// InsightFace 移动端推理类
/// 支持 ONNX Runtime 和 CoreML 模型推理
@available(iOS 13.0, *)
public class InsightFaceInference {
    
    // MARK: - Properties
    
    private var onnxSession: ORTSession?
    private var coreMLModel: MLModel?
    private var inputName: String = "input"
    private var outputNames: [String] = ["output"]
    private var inputShape: [NSNumber] = [1, 3, 640, 640]
    private var isONNXMode: Bool = true
    
    // MARK: - Initialization
    
    /// 初始化 InsightFace 推理器
    /// - Parameters:
    ///   - modelPath: 模型文件路径
    ///   - modelType: 模型类型 (.onnx 或 .mlmodel)
    ///   - inputName: 输入节点名称
    ///   - outputNames: 输出节点名称数组
    public init(modelPath: String, 
                modelType: ModelType = .onnx,
                inputName: String = "input",
                outputNames: [String] = ["output"]) throws {
        
        self.inputName = inputName
        self.outputNames = outputNames
        self.isONNXMode = (modelType == .onnx)
        
        try setupModel(modelPath: modelPath, modelType: modelType)
    }
    
    // MARK: - Model Setup
    
    private func setupModel(modelPath: String, modelType: ModelType) throws {
        switch modelType {
        case .onnx:
            try setupONNXModel(modelPath: modelPath)
        case .coreml:
            try setupCoreMLModel(modelPath: modelPath)
        }
    }
    
    private func setupONNXModel(modelPath: String) throws {
        // 初始化 ONNX Runtime
        let ortEnv = try ORTEnv(loggingLevel: .warning)
        let ortSessionOptions = try ORTSessionOptions()
        
        // 配置会话选项
        try ortSessionOptions.setIntraOpNumThreads(4)
        try ortSessionOptions.setGraphOptimizationLevel(.all)
        
        // 创建会话
        self.onnxSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: ortSessionOptions)
    }
    
    private func setupCoreMLModel(modelPath: String) throws {
        let modelURL = URL(fileURLWithPath: modelPath)
        self.coreMLModel = try MLModel(contentsOf: modelURL)
    }
    
    // MARK: - Public Methods
    
    /// 执行人脸检测
    /// - Parameter image: 输入图像
    /// - Returns: 检测结果
    public func detectFaces(in image: UIImage) throws -> [FaceDetection] {
        // 预处理图像
        let inputData = try preprocessImage(image)
        
        // 执行推理
        let outputs = try runInference(inputData: inputData)
        
        // 后处理结果
        return try postprocessDetections(outputs: outputs, imageSize: image.size)
    }
    
    /// 执行人脸识别
    /// - Parameter image: 输入图像
    /// - Returns: 人脸特征向量
    public func recognizeFace(in image: UIImage) throws -> [Float] {
        // 预处理图像
        let inputData = try preprocessImage(image)
        
        // 执行推理
        let outputs = try runInference(inputData: inputData)
        
        // 提取特征向量
        return try extractFaceFeatures(outputs: outputs)
    }
    
    /// 批量处理图像
    /// - Parameter images: 图像数组
    /// - Returns: 处理结果数组
    public func batchProcess(images: [UIImage]) throws -> [ProcessingResult] {
        var results: [ProcessingResult] = []
        
        for image in images {
            do {
                let detections = try detectFaces(in: image)
                let features = try recognizeFace(in: image)
                
                let result = ProcessingResult(
                    image: image,
                    detections: detections,
                    features: features
                )
                results.append(result)
            } catch {
                // 记录错误但继续处理其他图像
                print("处理图像失败: \(error.localizedDescription)")
                let errorResult = ProcessingResult(
                    image: image,
                    detections: [],
                    features: [],
                    error: error
                )
                results.append(errorResult)
            }
        }
        
        return results
    }
    
    // MARK: - Image Preprocessing
    
    private func preprocessImage(_ image: UIImage) throws -> [Float] {
        // 调整图像大小
        let targetSize = CGSize(width: inputShape[3].intValue, height: inputShape[2].intValue)
        let resizedImage = image.resized(to: targetSize)
        
        // 转换为像素数据
        guard let pixelData = resizedImage.pixelData() else {
            throw InferenceError.imageProcessingFailed
        }
        
        // 归一化到 [0, 1]
        let normalizedData = pixelData.map { Float($0) / 255.0 }
        
        // 转换为 NCHW 格式
        return convertToNCHW(normalizedData, 
                           width: Int(targetSize.width), 
                           height: Int(targetSize.height))
    }
    
    private func convertToNCHW(_ data: [Float], width: Int, height: Int) -> [Float] {
        let channels = 3
        var nchwData = [Float](repeating: 0, count: data.count)
        
        for c in 0..<channels {
            for h in 0..<height {
                for w in 0..<width {
                    let hwcIndex = h * width * channels + w * channels + c
                    let nchwIndex = c * height * width + h * width + w
                    nchwData[nchwIndex] = data[hwcIndex]
                }
            }
        }
        
        return nchwData
    }
    
    // MARK: - Inference
    
    private func runInference(inputData: [Float]) throws -> [[Float]] {
        if isONNXMode {
            return try runONNXInference(inputData: inputData)
        } else {
            return try runCoreMLInference(inputData: inputData)
        }
    }
    
    private func runONNXInference(inputData: [Float]) throws -> [[Float]] {
        guard let session = onnxSession else {
            throw InferenceError.sessionNotInitialized
        }
        
        // 创建输入张量
        let inputTensor = try ORTValue(tensorData: NSMutableData(data: Data(bytes: inputData, count: inputData.count * MemoryLayout<Float>.size)),
                                     elementType: .float,
                                     shape: inputShape)
        
        // 执行推理
        let outputs = try session.run(withInputs: [inputName: inputTensor],
                                    outputNames: Set(outputNames),
                                    runOptions: nil)
        
        // 提取输出数据
        var result: [[Float]] = []
        for outputName in outputNames {
            if let outputTensor = outputs[outputName] {
                let outputData = try outputTensor.tensorData()
                let floatData = outputData.withUnsafeBytes { bytes in
                    Array(bytes.bindMemory(to: Float.self))
                }
                result.append(floatData)
            }
        }
        
        return result
    }
    
    private func runCoreMLInference(inputData: [Float]) throws -> [[Float]] {
        guard let model = coreMLModel else {
            throw InferenceError.sessionNotInitialized
        }
        
        // 创建输入
        let inputArray = try MLMultiArray(shape: inputShape, dataType: .float32)
        for (index, value) in inputData.enumerated() {
            inputArray[index] = NSNumber(value: value)
        }
        
        let input = [inputName: inputArray]
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: input)
        
        // 执行推理
        let prediction = try model.prediction(from: inputProvider)
        
        // 提取输出
        var result: [[Float]] = []
        for outputName in outputNames {
            if let outputFeature = prediction.featureValue(for: outputName),
               let outputArray = outputFeature.multiArrayValue {
                let floatData = (0..<outputArray.count).map { Float(truncating: outputArray[$0]) }
                result.append(floatData)
            }
        }
        
        return result
    }
    
    // MARK: - Postprocessing
    
    private func postprocessDetections(outputs: [[Float]], imageSize: CGSize) throws -> [FaceDetection] {
        // 这里需要根据具体的模型输出格式来实现
        // 通常包括边界框、置信度、关键点等
        var detections: [FaceDetection] = []
        
        // 示例实现 - 需要根据实际模型调整
        if let detectionOutput = outputs.first {
            let numDetections = detectionOutput.count / 6 // 假设每个检测包含6个值
            
            for i in 0..<numDetections {
                let startIndex = i * 6
                let confidence = detectionOutput[startIndex + 4]
                
                if confidence > 0.5 { // 置信度阈值
                    let x = detectionOutput[startIndex] * Float(imageSize.width)
                    let y = detectionOutput[startIndex + 1] * Float(imageSize.height)
                    let width = detectionOutput[startIndex + 2] * Float(imageSize.width)
                    let height = detectionOutput[startIndex + 3] * Float(imageSize.height)
                    
                    let detection = FaceDetection(
                        boundingBox: CGRect(x: CGFloat(x), y: CGFloat(y), 
                                          width: CGFloat(width), height: CGFloat(height)),
                        confidence: confidence,
                        landmarks: [] // 需要根据模型输出添加关键点
                    )
                    detections.append(detection)
                }
            }
        }
        
        return detections
    }
    
    private func extractFaceFeatures(outputs: [[Float]]) throws -> [Float] {
        // 提取人脸特征向量
        guard let features = outputs.first else {
            throw InferenceError.featureExtractionFailed
        }
        
        // 归一化特征向量
        return normalizeFeatures(features)
    }
    
    private func normalizeFeatures(_ features: [Float]) -> [Float] {
        // L2 归一化
        let magnitude = sqrt(features.reduce(0) { $0 + $1 * $1 })
        return features.map { $0 / magnitude }
    }
}

// MARK: - Supporting Types

public enum ModelType {
    case onnx
    case coreml
}

public struct FaceDetection {
    public let boundingBox: CGRect
    public let confidence: Float
    public let landmarks: [CGPoint]
    
    public init(boundingBox: CGRect, confidence: Float, landmarks: [CGPoint] = []) {
        self.boundingBox = boundingBox
        self.confidence = confidence
        self.landmarks = landmarks
    }
}

public struct ProcessingResult {
    public let image: UIImage
    public let detections: [FaceDetection]
    public let features: [Float]
    public let error: Error?
    
    public init(image: UIImage, detections: [FaceDetection], features: [Float], error: Error? = nil) {
        self.image = image
        self.detections = detections
        self.features = features
        self.error = error
    }
}

public enum InferenceError: Error, LocalizedError {
    case sessionNotInitialized
    case imageProcessingFailed
    case featureExtractionFailed
    case modelLoadFailed
    
    public var errorDescription: String? {
        switch self {
        case .sessionNotInitialized:
            return "推理会话未初始化"
        case .imageProcessingFailed:
            return "图像处理失败"
        case .featureExtractionFailed:
            return "特征提取失败"
        case .modelLoadFailed:
            return "模型加载失败"
        }
    }
}

// MARK: - UIImage Extensions

extension UIImage {
    func resized(to size: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext() ?? self
    }
    
    func pixelData() -> [UInt8]? {
        guard let cgImage = cgImage else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        guard let context = CGContext(data: &pixelData,
                                    width: width,
                                    height: height,
                                    bitsPerComponent: bitsPerComponent,
                                    bytesPerRow: bytesPerRow,
                                    space: CGColorSpaceCreateDeviceRGB(),
                                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return pixelData
    }
}
