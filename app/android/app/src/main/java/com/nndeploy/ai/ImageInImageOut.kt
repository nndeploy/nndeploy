// /home/always/github/public/nndeploy/app/android/app/src/main/java/com/nndeploy/ai/ImageInImageOut.kt
package com.nndeploy.ai

import android.content.Context
import android.net.Uri
import android.util.Log
import com.nndeploy.dag.GraphRunner
import com.nndeploy.base.FileUtils
import java.io.File

/**
 * AI算法处理器
 */
object ImageInImageOut {
    
    /**
     * 分割算法处理
     */
    suspend fun processSegmentation(context: Context, imageUri: Uri): ProcessResult {
        return try {
            Log.w("ImageInImageOut", "Starting segmentation processing")
            
            // 1) 确保外部资源就绪
            val extResDir = FileUtils.ensureExternalResourcesReady(context)
            val extRoot = FileUtils.getExternalRoot(context)
            val extWorkflowDir = File(extRoot, "workflow").apply { mkdirs() }

            // 2) 读取 assets 的 workflow，并把相对路径替换为外部绝对路径
            val workflowAsset = "resources/workflow/SegmentationDemo.json"
            val rawJson = context.assets.open(workflowAsset).bufferedReader().use { it.readText() }
            val resolvedJson = rawJson.replace("resources/", "${extResDir.absolutePath}/".replace("\\", "/"))
            
            // 3) 写到外部私有目录，得到真实文件路径
            val workflowOut = File(extWorkflowDir, "SegmentationDemo_resolved.json").apply {
                writeText(resolvedJson)
            }

            // 4) 以文件路径运行底层
            val runner = GraphRunner()
            runner.setJsonFile(true)
            runner.setTimeProfile(true)
            runner.setDebug(true)
            
            val ok = runner.run(workflowOut.absolutePath, "seg_demo", "task_${System.currentTimeMillis()}")
            runner.close()
            
            // 获取结果图片路径
            val resultImagePath = File(extResDir, "images/result.segment.jpg")
            if (resultImagePath.exists()) {
                ProcessResult.Success(Uri.fromFile(resultImagePath))
            } else {
                ProcessResult.Error("分割结果图片未找到")
            }
            
        } catch (e: Exception) {
            Log.e("ImageInImageOut", "Segmentation processing failed", e)
            ProcessResult.Error("分割处理失败: ${e.message}")
        }
    }
}

/**
 * 处理结果封装
 */
sealed class ProcessResult {
    data class Success(val resultUri: Uri) : ProcessResult()
    data class Error(val message: String) : ProcessResult()
}