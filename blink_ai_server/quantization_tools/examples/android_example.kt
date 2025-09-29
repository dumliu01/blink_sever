package com.insightface.mobile.example

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.insightface.mobile.InsightFaceInference
import com.insightface.mobile.FaceDetection
import com.insightface.mobile.ProcessingResult
import java.io.IOException

/**
 * Android InsightFace 移动端推理示例
 */
class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "InsightFaceExample"
        private const val REQUEST_PERMISSIONS = 1001
    }
    
    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var processButton: Button
    private lateinit var batchProcessButton: Button
    private lateinit var performanceTestButton: Button
    
    private var insightFace: InsightFaceInference? = null
    private var currentBitmap: Bitmap? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initViews()
        setupInsightFace()
        checkPermissions()
    }
    
    private fun initViews() {
        imageView = findViewById(R.id.imageView)
        resultTextView = findViewById(R.id.resultTextView)
        processButton = findViewById(R.id.processButton)
        batchProcessButton = findViewById(R.id.batchProcessButton)
        performanceTestButton = findViewById(R.id.performanceTestButton)
        
        processButton.setOnClickListener { selectImage() }
        batchProcessButton.setOnClickListener { batchProcessExample() }
        performanceTestButton.setOnClickListener { performanceTestExample() }
    }
    
    private fun setupInsightFace() {
        try {
            // 初始化 InsightFace 推理器
            // 这里需要将量化后的模型文件添加到 assets 目录
            insightFace = InsightFaceInference(
                context = this,
                modelPath = "buffalo_l_int8.tflite", // 或 .onnx 文件
                modelType = InsightFaceInference.ModelType.TFLITE,
                inputName = "input",
                outputNames = listOf("output"),
                inputShape = intArrayOf(1, 3, 640, 640)
            )
            
            Log.i(TAG, "InsightFace 初始化成功")
            Toast.makeText(this, "InsightFace 初始化成功", Toast.LENGTH_SHORT).show()
            
        } catch (e: Exception) {
            Log.e(TAG, "InsightFace 初始化失败", e)
            Toast.makeText(this, "模型初始化失败: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
    
    private fun checkPermissions() {
        val permissions = arrayOf(
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.CAMERA
        )
        
        val permissionsToRequest = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        
        if (permissionsToRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, permissionsToRequest.toTypedArray(), REQUEST_PERMISSIONS)
        }
    }
    
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == REQUEST_PERMISSIONS) {
            val allGranted = grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            if (!allGranted) {
                Toast.makeText(this, "需要权限才能选择图像", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    private fun selectImage() {
        // 这里应该实现图像选择逻辑
        // 为了演示，我们使用一个示例图像
        loadSampleImage()
    }
    
    private fun loadSampleImage() {
        try {
            // 从 assets 加载示例图像
            val inputStream = assets.open("sample_face.jpg")
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream.close()
            
            currentBitmap = bitmap
            imageView.setImageBitmap(bitmap)
            
            // 自动处理图像
            processImage(bitmap)
            
        } catch (e: IOException) {
            Log.e(TAG, "加载示例图像失败", e)
            Toast.makeText(this, "加载示例图像失败", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun processImage(bitmap: Bitmap) {
        val insightFace = this.insightFace ?: run {
            Toast.makeText(this, "InsightFace 未初始化", Toast.LENGTH_SHORT).show()
            return
        }
        
        // 在后台线程执行推理
        Thread {
            try {
                // 执行人脸检测
                val detections = insightFace.detectFaces(bitmap)
                
                // 执行人脸识别
                val features = insightFace.recognizeFace(bitmap)
                
                // 在主线程更新UI
                runOnUiThread {
                    updateUI(detections, features)
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "图像处理失败", e)
                runOnUiThread {
                    Toast.makeText(this, "处理失败: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }
    
    private fun updateUI(detections: List<FaceDetection>, features: FloatArray) {
        // 更新图像显示（绘制检测框）
        val bitmapWithDetections = drawDetections(currentBitmap ?: return, detections)
        imageView.setImageBitmap(bitmapWithDetections)
        
        // 更新结果文本
        val resultText = buildString {
            appendLine("检测到 ${detections.size} 个人脸")
            appendLine()
            
            detections.forEachIndexed { index, detection ->
                appendLine("人脸 ${index + 1}:")
                appendLine("  置信度: ${String.format("%.2f", detection.confidence)}")
                appendLine("  位置: ${detection.boundingBox}")
                appendLine()
            }
            
            if (features.isNotEmpty()) {
                appendLine("特征向量维度: ${features.size}")
            }
        }
        
        resultTextView.text = resultText
    }
    
    private fun drawDetections(bitmap: Bitmap, detections: List<FaceDetection>): Bitmap {
        val resultBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(resultBitmap)
        val paint = Paint().apply {
            color = android.graphics.Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }
        
        val textPaint = Paint().apply {
            color = android.graphics.Color.RED
            textSize = 32f
        }
        
        detections.forEach { detection ->
            // 绘制边界框
            canvas.drawRect(detection.boundingBox, paint)
            
            // 绘制置信度
            val confidenceText = String.format("%.2f", detection.confidence)
            canvas.drawText(
                confidenceText,
                detection.boundingBox.left,
                detection.boundingBox.top - 10,
                textPaint
            )
        }
        
        return resultBitmap
    }
    
    private fun batchProcessExample() {
        val insightFace = this.insightFace ?: run {
            Toast.makeText(this, "InsightFace 未初始化", Toast.LENGTH_SHORT).show()
            return
        }
        
        // 创建示例图像列表
        val images = mutableListOf<Bitmap>()
        try {
            for (i in 1..5) {
                val inputStream = assets.open("sample_face_$i.jpg")
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream.close()
                images.add(bitmap)
            }
        } catch (e: IOException) {
            Log.e(TAG, "加载批量图像失败", e)
            Toast.makeText(this, "加载批量图像失败", Toast.LENGTH_SHORT).show()
            return
        }
        
        // 在后台线程执行批量处理
        Thread {
            try {
                val results = insightFace.batchProcess(images)
                
                runOnUiThread {
                    handleBatchResults(results)
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "批量处理失败", e)
                runOnUiThread {
                    Toast.makeText(this, "批量处理失败: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }
    
    private fun handleBatchResults(results: List<ProcessingResult>) {
        var successCount = 0
        var errorCount = 0
        
        results.forEach { result ->
            if (result.error == null) {
                successCount++
                Log.i(TAG, "处理成功: 检测到 ${result.detections.size} 个人脸")
            } else {
                errorCount++
                Log.e(TAG, "处理失败: ${result.error?.message}")
            }
        }
        
        Toast.makeText(
            this,
            "批量处理完成\n成功: $successCount, 失败: $errorCount",
            Toast.LENGTH_LONG
        ).show()
    }
    
    private fun performanceTestExample() {
        val insightFace = this.insightFace ?: run {
            Toast.makeText(this, "InsightFace 未初始化", Toast.LENGTH_SHORT).show()
            return
        }
        
        val testImage = currentBitmap ?: run {
            Toast.makeText(this, "请先选择图像", Toast.LENGTH_SHORT).show()
            return
        }
        
        val iterations = 100
        val times = mutableListOf<Long>()
        
        // 在后台线程执行性能测试
        Thread {
            try {
                for (i in 0 until iterations) {
                    val startTime = System.currentTimeMillis()
                    
                    insightFace.detectFaces(testImage)
                    
                    val endTime = System.currentTimeMillis()
                    times.add(endTime - startTime)
                    
                    if (i % 10 == 0) {
                        Log.i(TAG, "性能测试进度: $i/$iterations")
                    }
                }
                
                val averageTime = times.average()
                val fps = 1000.0 / averageTime
                
                runOnUiThread {
                    Toast.makeText(
                        this,
                        "性能测试结果\n平均推理时间: ${String.format("%.2f", averageTime)}ms\n推理速度: ${String.format("%.2f", fps)} FPS",
                        Toast.LENGTH_LONG
                    ).show()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "性能测试失败", e)
                runOnUiThread {
                    Toast.makeText(this, "性能测试失败: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        insightFace?.release()
    }
}
