package com.nndeploy.ai

import android.content.Context
import android.util.Log
import com.nndeploy.dag.GraphRunner
import com.nndeploy.base.FileUtils
import java.io.File

/**
 * AI Algorithm Processor - Supports prompt input and text output
 */
object PromptInPromptOut {
    
    // Store conversation history Map, key is session ID, value is conversation history
    private val conversationHistory = mutableMapOf<String, MutableList<ConversationMessage>>()
    
    /**
     * Conversation message data class
     */
    data class ConversationMessage(
        val role: String, // "user" or "assistant"
        val content: String,
        val timestamp: Long = System.currentTimeMillis()
    )
    
    /**
     * Processing result wrapper
     */
    sealed class PromptProcessResult {
        data class Success(val response: String, val conversationId: String) : PromptProcessResult()
        data class Error(val message: String) : PromptProcessResult()
    }
    
    /**
     * Intelligent conversation processing - Supports multi-turn dialogue
     */
    suspend fun processPromptInPromptOut(
        context: Context, 
        prompt: String, 
        alg: AIAlgorithm,
        conversationId: String = "default"
    ): PromptProcessResult {
        return try {
            Log.w("PromptInPromptOut", "Starting processing for ${alg.name}")
            
            // 1) Ensure external resources are ready
            val extResDir = FileUtils.ensureExternalResourcesReady(context)
            val extRoot = FileUtils.getExternalRoot(context)
            val extWorkflowDir = File(extResDir, "workflow").apply { mkdirs() }
            
            // Print three variables
            Log.d("PromptInPromptOut", "extResDir: ${extResDir.absolutePath}")
            Log.d("PromptInPromptOut", "extRoot: ${extRoot.absolutePath}")
            Log.d("PromptInPromptOut", "extWorkflowDir: ${extWorkflowDir.absolutePath}")

            val workflowAsset = alg.workflowAsset
            
            // 2) Get or create conversation history
            val history = conversationHistory.getOrPut(conversationId) { mutableListOf() }
            
            // 3) Build complete prompt including history
            // val fullPrompt = buildFullPrompt(prompt, history)
            // Log.d("PromptInPromptOut", "Full prompt with history: $fullPrompt")
            
            // 4) Read workflow from assets and replace relative paths with external absolute paths
            val rawJson = context.assets.open(workflowAsset).bufferedReader().use { it.readText() }
            val resolvedJson = rawJson.replace("resources/", "${extResDir.absolutePath}/".replace("\\", "/"))
            // Print resolved JSON content
            Log.d("PromptInPromptOut", "Resolved JSON content: $resolvedJson")
            
            // 5) Write to external private directory, get real file path
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
            
            // Set input prompt
            // Log.d("PromptInPromptOut", "prompt: $prompt")
            runner.setNodeValue(input_node_param.keys.first(), input_node_param.values.first(), prompt)
            
            // Set output path
            val resultPath = File(extResDir, "text/result.${alg.id}.${System.currentTimeMillis()}.txt")
            resultPath.parentFile?.mkdirs()
            runner.setNodeValue(output_node_param.keys.first(), output_node_param.values.first(), resultPath.absolutePath)
            
            val ok = runner.run(workflowOut.absolutePath, alg.id, "task_${System.currentTimeMillis()}")
            runner.close()
            
            // Log.d("PromptInPromptOut", "resultPath: ${resultPath.absolutePath}")
            if (resultPath.exists()) {
                Log.d("PromptInPromptOut", "resultPath exists")
                val response = resultPath.readText().trim()
                
                // 7) Update conversation history
                history.add(ConversationMessage("user", prompt))
                history.add(ConversationMessage("assistant", response))
                
                // 8) Limit history length to avoid excessive memory usage
                if (history.size > 20) { // Keep recent 10 rounds of conversation
                    history.removeAt(0)
                    history.removeAt(0)
                }
                
                PromptProcessResult.Success(response, conversationId)
            } else {
                Log.d("PromptInPromptOut", "resultPath not exists")
                PromptProcessResult.Error("Result file not found: ${resultPath.absolutePath}")
            }
            
        } catch (e: Exception) {
            Log.e("PromptInPromptOut", "Prompt processing failed", e)
            PromptProcessResult.Error("Processing failed: ${e.message}")
        }
    }
    
    /**
     * Build complete prompt including history
     */
    private fun buildFullPrompt(currentPrompt: String, history: List<ConversationMessage>): String {
        if (history.isEmpty()) {
            return currentPrompt
        }
        
        val contextBuilder = StringBuilder()
        contextBuilder.append("The following is conversation history:\n")
        
        // Add conversation history
        history.takeLast(10).forEach { message -> // Only take recent 5 rounds of conversation
            when (message.role) {
                "user" -> contextBuilder.append("User: ${message.content}\n")
                "assistant" -> contextBuilder.append("Assistant: ${message.content}\n")
            }
        }
        
        contextBuilder.append("\nCurrent user question: $currentPrompt")
        contextBuilder.append("\nPlease answer the current question based on conversation history:")
        
        return contextBuilder.toString()
    }
    
    /**
     * Clear conversation history for specified session
     */
    fun clearConversationHistory(conversationId: String = "default") {
        conversationHistory.remove(conversationId)
        Log.d("PromptInPromptOut", "Cleared conversation history for: $conversationId")
    }
    
    /**
     * Get conversation history for specified session
     */
    fun getConversationHistory(conversationId: String = "default"): List<ConversationMessage> {
        return conversationHistory[conversationId]?.toList() ?: emptyList()
    }
    
    /**
     * Get all conversation IDs
     */
    fun getAllConversationIds(): Set<String> {
        return conversationHistory.keys.toSet()
    }
}
