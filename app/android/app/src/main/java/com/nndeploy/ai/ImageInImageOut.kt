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
 * AI Algorithm Processor - supports image, video, camera input
 */
object ImageInImageOut {
    
    /**
     * Segmentation algorithm processing
     */
    suspend fun processImageInImageOut(context: Context, inputUri: Uri, alg: AIAlgorithm): ProcessResult {
        return try {
            Log.w("ImageInImageOut", "Starting processing for ${alg.name}")
            
            // 1) Ensure external resources are ready
            val extResDir = FileUtils.ensureExternalResourcesReady(context)
            val extRoot = FileUtils.getExternalRoot(context)
            val extWorkflowDir = File(extResDir, "workflow").apply { mkdirs() }
            
            // Print three variables
            Log.d("ImageInImageOut", "extResDir: ${extResDir.absolutePath}")
            Log.d("ImageInImageOut", "extRoot: ${extRoot.absolutePath}")
            Log.d("ImageInImageOut", "extWorkflowDir: ${extWorkflowDir.absolutePath}")

            val workflowAsset = alg.workflowAsset
            
            // 3) Preprocess input data
            val (processedInputFile, processedInputUri) = ImageUtils.preprocessImage(context, inputUri)
            
            // 4) Read workflow from assets and replace relative paths with external absolute paths
            val rawJson = context.assets.open(workflowAsset).bufferedReader().use { it.readText() }
            val resolvedJson = rawJson.replace("resources/", "${extResDir.absolutePath}/".replace("\\", "/"))
            // Print resolved JSON content
            Log.d("ImageInImageOut", "Resolved JSON content: $resolvedJson")
            
            // 5) Write to external private directory to get real file path
            val workflowOut = File(extWorkflowDir, alg.id + "_resolved.json").apply {
                writeText(resolvedJson)
            }

            // 6) Run underlying system with file path
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
                ProcessResult.Error("Result file not found: ${resultPath.absolutePath}")
            }
            
        } catch (e: Exception) {
            Log.e("ImageInImageOut", "Segmentation processing failed", e)
            ProcessResult.Error("Processing failed: ${e.message}")
        }
    }  
}

/**
 * Processing result wrapper
 */
sealed class ProcessResult {
    data class Success(val resultUri: Uri) : ProcessResult()
    data class Error(val message: String) : ProcessResult()
}
