// /home/always/github/public/nndeploy/app/android/app/src/main/java/com/nndeploy/ai/ImageInImageOut.kt
package com.nndeploy.ai

import android.content.Context
import android.net.Uri
import android.util.Log
import com.nndeploy.dag.GraphRunner
import com.nndeploy.base.FileUtils
import com.nndeploy.base.ImageUtils
import com.nndeploy.base.VideoUtils
import java.io.File

/**
 * AI算法处理器 - 支持图片、视频、摄像头输入
 */
object ImageInImageOut {
    
    /**
     * 分割算法处理
     */
    suspend fun processImageInImageOut(context: Context, inputUri: Uri, inputType: InputMediaType = InputMediaType.IMAGE): ProcessResult {
        return try {
            Log.w("ImageInImageOut", "Starting segmentation processing for ${inputType.name}")
            
            // 1) 确保外部资源就绪
            val extResDir = FileUtils.ensureExternalResourcesReady(context)
            val extRoot = FileUtils.getExternalRoot(context)
            val extWorkflowDir = File(extResDir, "workflow").apply { mkdirs() }
            
            // 打印三个变量
            Log.d("ImageInImageOut", "extResDir: ${extResDir.absolutePath}")
            Log.d("ImageInImageOut", "extRoot: ${extRoot.absolutePath}")
            Log.d("ImageInImageOut", "extWorkflowDir: ${extWorkflowDir.absolutePath}")

            // val workflowAsset = extWorkflowDir.absolutePath + "/ClassificationResNetMnn.json"
            val workflowAsset = "resources/workflow/ClassificationResNetMnn.json"
            
            // 3) 预处理输入数据
            val processedInputUri = ImageUtils.preprocessImage(context, inputUri)
            
            // 4) 读取 assets 的 workflow，并把相对路径替换为外部绝对路径
            val rawJson = context.assets.open(workflowAsset).bufferedReader().use { it.readText() }
            val resolvedJson = rawJson.replace("resources/", "${extResDir.absolutePath}/".replace("\\", "/"))
            // 打印解析后的JSON内容
            Log.d("ImageInImageOut", "Resolved JSON content: $resolvedJson")
            
            // 5) 写到外部私有目录，得到真实文件路径
            val workflowOut = File(extWorkflowDir, "Segmentation_${inputType.name}_resolved.json").apply {
                writeText(resolvedJson)
            }

            // 6) 以文件路径运行底层
            val runner = GraphRunner()
            runner.setJsonFile(true)
            runner.setTimeProfile(true)
            runner.setDebug(true)
            
            // 设置输入文件路径
            // runner.setNodeValue("input_node", "input_path", processedInputUri.path ?: "")
            
            val ok = runner.run(workflowOut.absolutePath, "seg_demo", "task_${System.currentTimeMillis()}")
            runner.close()
            
            // 7) 获取结果路径
            // val resultPath = getResultPath(extResDir, inputType, "segment")
            val resultPath = File(extResDir, "images/result.resnet.jpg")
            Log.d("ImageInImageOut", "resultPath: ${resultPath.absolutePath}")
            if (resultPath.exists()) {
                Log.d("ImageInImageOut", "resultPath exists")
                ProcessResult.Success(Uri.fromFile(resultPath))
            } else {
                Log.d("ImageInImageOut", "resultPath not exists")
                ProcessResult.Error("分割结果文件未找到: ${resultPath.absolutePath}")
            }
            
        } catch (e: Exception) {
            Log.e("ImageInImageOut", "Segmentation processing failed", e)
            ProcessResult.Error("分割处理失败: ${e.message}")
        }
    }  
}

/**
 * 输入媒体类型枚举
 */
enum class InputMediaType {
    IMAGE,          // 图片
    VIDEO,          // 视频
    CAMERA_PHOTO,   // 摄像头拍照
    CAMERA_VIDEO    // 摄像头录像
}

/**
 * 处理结果封装
 */
sealed class ProcessResult {
    data class Success(val resultUri: Uri) : ProcessResult()
    data class Error(val message: String) : ProcessResult()
}

/**
 * 获取处理结果文件路径
 * @param extResDir 外部结果目录
 * @param inputType 输入媒体类型
 * @param taskType 任务类型（如"segment"）
 * @return 结果文件路径
 */
private fun getResultPath(extResDir: File, inputType: InputMediaType, taskType: String): File {
    return when (inputType) {
        InputMediaType.IMAGE, InputMediaType.CAMERA_PHOTO -> {
            // 图片类型的结果文件
            File(extResDir, "${taskType}_result.jpg")
        }
        InputMediaType.VIDEO, InputMediaType.CAMERA_VIDEO -> {
            // 视频类型的结果文件
            File(extResDir, "${taskType}_result.mp4")
        }
    }
}
