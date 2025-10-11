package com.nndeploy.ai

import android.content.Context
import android.util.Log
import com.nndeploy.dag.GraphRunner
import com.nndeploy.base.FileUtils
import java.io.File

/**
 * AI算法处理器 - 支持提示词输入和文本输出
 */
object PromptInPromptOut {
    
    // 存储对话历史的Map，key为会话ID，value为对话历史
    private val conversationHistory = mutableMapOf<String, MutableList<ConversationMessage>>()
    
    /**
     * 对话消息数据类
     */
    data class ConversationMessage(
        val role: String, // "user" 或 "assistant"
        val content: String,
        val timestamp: Long = System.currentTimeMillis()
    )
    
    /**
     * 处理结果封装
     */
    sealed class PromptProcessResult {
        data class Success(val response: String, val conversationId: String) : PromptProcessResult()
        data class Error(val message: String) : PromptProcessResult()
    }
    
    /**
     * 智能对话处理 - 支持多轮对话
     */
    suspend fun processPromptInPromptOut(
        context: Context, 
        prompt: String, 
        alg: AIAlgorithm,
        conversationId: String = "default"
    ): PromptProcessResult {
        return try {
            Log.w("PromptInPromptOut", "Starting processing for ${alg.name}")
            
            // 1) 确保外部资源就绪
            val extResDir = FileUtils.ensureExternalResourcesReady(context)
            val extRoot = FileUtils.getExternalRoot(context)
            val extWorkflowDir = File(extResDir, "workflow").apply { mkdirs() }
            
            // 打印三个变量
            Log.d("PromptInPromptOut", "extResDir: ${extResDir.absolutePath}")
            Log.d("PromptInPromptOut", "extRoot: ${extRoot.absolutePath}")
            Log.d("PromptInPromptOut", "extWorkflowDir: ${extWorkflowDir.absolutePath}")

            val workflowAsset = alg.workflowAsset
            
            // 2) 获取或创建对话历史
            val history = conversationHistory.getOrPut(conversationId) { mutableListOf() }
            
            // 3) 构建包含历史的完整提示词
            val fullPrompt = buildFullPrompt(prompt, history)
            Log.d("PromptInPromptOut", "Full prompt with history: $fullPrompt")
            
            // 4) 读取 assets 的 workflow，并把相对路径替换为外部绝对路径
            val rawJson = context.assets.open(workflowAsset).bufferedReader().use { it.readText() }
            val resolvedJson = rawJson.replace("resources/", "${extResDir.absolutePath}/".replace("\\", "/"))
            // 打印解析后的JSON内容
            Log.d("PromptInPromptOut", "Resolved JSON content: $resolvedJson")
            
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
            
            // 设置输入提示词
            runner.setNodeValue(input_node_param.keys.first(), input_node_param.values.first(), fullPrompt)
            
            // 设置输出路径
            val resultPath = File(extResDir, "text/result.${alg.id}.${System.currentTimeMillis()}.txt")
            resultPath.parentFile?.mkdirs()
            runner.setNodeValue(output_node_param.keys.first(), output_node_param.values.first(), resultPath.absolutePath)
            
            val ok = runner.run(workflowOut.absolutePath, alg.id, "task_${System.currentTimeMillis()}")
            runner.close()
            
            Log.d("PromptInPromptOut", "resultPath: ${resultPath.absolutePath}")
            if (resultPath.exists()) {
                Log.d("PromptInPromptOut", "resultPath exists")
                val response = resultPath.readText().trim()
                
                // 7) 更新对话历史
                history.add(ConversationMessage("user", prompt))
                history.add(ConversationMessage("assistant", response))
                
                // 8) 限制历史长度，避免内存过大
                if (history.size > 20) { // 保留最近10轮对话
                    history.removeAt(0)
                    history.removeAt(0)
                }
                
                PromptProcessResult.Success(response, conversationId)
            } else {
                Log.d("PromptInPromptOut", "resultPath not exists")
                PromptProcessResult.Error("结果文件未找到: ${resultPath.absolutePath}")
            }
            
        } catch (e: Exception) {
            Log.e("PromptInPromptOut", "Prompt processing failed", e)
            PromptProcessResult.Error("处理失败: ${e.message}")
        }
    }
    
    /**
     * 构建包含历史的完整提示词
     */
    private fun buildFullPrompt(currentPrompt: String, history: List<ConversationMessage>): String {
        if (history.isEmpty()) {
            return currentPrompt
        }
        
        val contextBuilder = StringBuilder()
        contextBuilder.append("以下是对话历史:\n")
        
        // 添加历史对话
        history.takeLast(10).forEach { message -> // 只取最近5轮对话
            when (message.role) {
                "user" -> contextBuilder.append("用户: ${message.content}\n")
                "assistant" -> contextBuilder.append("助手: ${message.content}\n")
            }
        }
        
        contextBuilder.append("\n当前用户问题: $currentPrompt")
        contextBuilder.append("\n请基于对话历史回答当前问题:")
        
        return contextBuilder.toString()
    }
    
    /**
     * 清除指定会话的对话历史
     */
    fun clearConversationHistory(conversationId: String = "default") {
        conversationHistory.remove(conversationId)
        Log.d("PromptInPromptOut", "Cleared conversation history for: $conversationId")
    }
    
    /**
     * 获取指定会话的对话历史
     */
    fun getConversationHistory(conversationId: String = "default"): List<ConversationMessage> {
        return conversationHistory[conversationId]?.toList() ?: emptyList()
    }
    
    /**
     * 获取所有会话ID
     */
    fun getAllConversationIds(): Set<String> {
        return conversationHistory.keys.toSet()
    }
}
