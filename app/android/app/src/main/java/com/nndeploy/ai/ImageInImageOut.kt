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
    suspend fun processImageInImageOut(context: Context, inputUri: Uri, alg: AIAlgorithm): ProcessResult {
        return try {
            Log.w("ImageInImageOut", "Starting processing for ${alg.name}")
            
            // 1) 确保外部资源就绪
            val extResDir = FileUtils.ensureExternalResourcesReady(context)
            val extRoot = FileUtils.getExternalRoot(context)
            val extWorkflowDir = File(extResDir, "workflow").apply { mkdirs() }
            
            // 打印三个变量
            Log.d("ImageInImageOut", "extResDir: ${extResDir.absolutePath}")
            Log.d("ImageInImageOut", "extRoot: ${extRoot.absolutePath}")
            Log.d("ImageInImageOut", "extWorkflowDir: ${extWorkflowDir.absolutePath}")

            val workflowAsset = alg.workflowAsset
            
            // 3) 预处理输入数据
            val (processedInputFile, processedInputUri) = ImageUtils.preprocessImage(context, inputUri)
            
            // 4) 读取 assets 的 workflow，并把相对路径替换为外部绝对路径
            val rawJson = context.assets.open(workflowAsset).bufferedReader().use { it.readText() }
            val resolvedJson = rawJson.replace("resources/", "${extResDir.absolutePath}/".replace("\\", "/"))
            // 打印解析后的JSON内容
            Log.d("ImageInImageOut", "Resolved JSON content: $resolvedJson")
            
            // 5) 写到外部私有目录，得到真实文件路径
            val workflowOut = File(extWorkflowDir, alg.id + "_resolved.json").apply {
                writeText(resolvedJson)
            }

            // 6) 以文件路径运行底层
            val runner = GraphRunner()
            runner.setJsonFile(true)
            runner.setTimeProfile(true)
            runner.setDebug(true)
            val input_node_param = alg.parameters["input_node"] as Map<String, String>
            val output_node_param = alg.parameters["output_node"] as Map<String, String>
            runner.setNodeValue(input_node_param.keys.first(), input_node_param.values.first(), processedInputFile.absolutePath)
            val resultPath = File(extResDir, "images/result.${alg.id}.jpg")

            runner.setNodeValue(output_node_param.keys.first(), output_node_param.values.first(), resultPath.absolutePath)
            
            val ok = runner.run(workflowOut.absolutePath, alg.id, "task_${System.currentTimeMillis()}")
            runner.close()
            
            Log.d("ImageInImageOut", "resultPath: ${resultPath.absolutePath}")
            if (resultPath.exists()) {
                Log.d("ImageInImageOut", "resultPath exists")
                ProcessResult.Success(Uri.fromFile(resultPath))
            } else {
                Log.d("ImageInImageOut", "resultPath not exists")
                ProcessResult.Error("结果文件未找到: ${resultPath.absolutePath}")
            }
            
        } catch (e: Exception) {
            Log.e("ImageInImageOut", "Segmentation processing failed", e)
            ProcessResult.Error("处理失败: ${e.message}")
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

