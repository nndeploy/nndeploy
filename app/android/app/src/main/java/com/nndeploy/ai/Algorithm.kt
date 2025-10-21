package com.nndeploy.ai

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.ui.graphics.vector.ImageVector

/**
 * AI Algorithm Information Data Class
 */
data class AIAlgorithm(
    val id: String,
    val name: String,
    val description: String,
    val icon: ImageVector,
    val inputType: List<InOutType>,
    val outputType: List<InOutType>,
    val category: String,
    val workflowAsset: String,
    val tags: List<String>,
    val version: String = "1.0.0",
    val author: String = "nndeploy",
    val parameters: Map<String, Any> = emptyMap(), // Algorithm parameters
    val processFunction: String = "none"
)

/**
 * Input/Output Type Enum
 */
enum class InOutType(val displayName: String, val icon: ImageVector) {
    IMAGE("Image", Icons.Default.Image),
    VIDEO("Video", Icons.Default.VideoFile),
    CAMERA("Camera", Icons.Default.CameraAlt),
    PROMPT("Prompt", Icons.Default.TextFields),
    AUDIO("Audio", Icons.Default.AudioFile),
    TEXT("Text", Icons.Default.Description),
    ALL("All Supported", Icons.Default.AllInclusive);
    
    companion object {
        fun fromString(type: String): InOutType? {
            return values().find { it.name.equals(type, ignoreCase = true) }
        }
    }
}

/**
 * Input Media Type Enum
 */
enum class InputMediaType {
    IMAGE,
    VIDEO,
    CAMERA,
    AUDIO,
    TEXT
}

/**
 * Algorithm Category Enum
 */
enum class AlgorithmCategory(val displayName: String, val icon: ImageVector) {
    COMPUTER_VISION("Computer Vision", Icons.Default.Visibility),
    NATURAL_LANGUAGE("Natural Language Processing", Icons.Default.TextFields),
    AUDIO_PROCESSING("Audio Processing", Icons.Default.AudioFile),
    GENERATIVE_AI("Generative AI", Icons.Default.AutoAwesome),
    OTHER("Other", Icons.Default.Category)
}

/**
 * Algorithm Processing Status
 */
enum class ProcessingStatus {
    IDLE,           // Idle
    LOADING,        // Loading
    PROCESSING,     // Processing
    COMPLETED,      // Completed
    ERROR,          // Error
    CANCELLED       // Cancelled
}

/**
 * Algorithm Runtime Information
 */
data class AlgorithmRuntime(
    val status: ProcessingStatus = ProcessingStatus.IDLE,
    val progress: Float = 0f,           // Processing progress 0.0-1.0
    val startTime: Long = 0L,           // Start time
    val endTime: Long = 0L,             // End time
    val errorMessage: String? = null
)

/**
 * Algorithm Factory Class
 */
object AlgorithmFactory {
    
    /**
     * Create predefined algorithm list
     */
    fun createDefaultAlgorithms(): List<AIAlgorithm> {
        return listOf(
            // Computer Vision Algorithms
            AIAlgorithm(
                id = "image_segmentation",
                name = "nndeploy Segment",
                description = "Intelligently identify and segment different objects and regions in images",
                icon = Icons.Default.Crop,
                inputType = listOf(InOutType.IMAGE),
                outputType = listOf(InOutType.IMAGE),
                category = AlgorithmCategory.COMPUTER_VISION.displayName,
                workflowAsset = "resources/workflow/SegmentRMBGMNN.json",
                tags = listOf("segmentation", "object detection", "image processing"),
                parameters = mapOf(
                    "input_node" to mapOf("OpenCvImageDecode_11" to "path_"),
                    "output_node" to mapOf("OpenCvImageEncode_16" to "path_"),
                ),
                processFunction = "processImageInImageOut"
            ),
            AIAlgorithm(
                id = "image_classification",
                name = "nndeploy Classification",
                description = "Intelligently identify object categories and labels in images",
                icon = Icons.Default.Category,
                inputType = listOf(InOutType.IMAGE),
                outputType = listOf(InOutType.TEXT),
                category = AlgorithmCategory.COMPUTER_VISION.displayName,
                workflowAsset = "resources/workflow/ClassificationResNetMnn.json",
                tags = listOf("classification", "recognition", "labeling"),
                parameters = mapOf(
                    "input_node" to mapOf("OpenCvImageDecode_11" to "path_"),
                    "output_node" to mapOf("OpenCvImageEncode_26" to "path_"),
                ),
                processFunction = "processImageInImageOut"
            ),
            
            // Natural Language Processing Algorithms
            AIAlgorithm(
                id = "text_chat",
                name = "nndeploy Chat",
                description = "Intelligent dialogue system based on large language models, supporting multi-turn conversations and context understanding",
                icon = Icons.Default.QuestionAnswer,
                inputType = listOf(InOutType.PROMPT),
                outputType = listOf(InOutType.TEXT),
                category = AlgorithmCategory.NATURAL_LANGUAGE.displayName,
                workflowAsset = "resources/workflow/QwenMNN.json",
                tags = listOf("dialogue", "chat", "Q&A"),
                parameters = mapOf(
                    "input_node" to mapOf("Prompt_4" to "user_content_"),
                    "output_node" to mapOf("LlmOut_3" to "path_"),
                ),
                processFunction = "processPromptInPromptOut"
            ),
            
            // Generative AI Algorithms
            // AIAlgorithm(
            //     id = "text_to_image",
            //     name = "Text to Image",
            //     description = "Generate corresponding image content based on text descriptions, supporting multiple artistic styles",
            //     icon = Icons.Default.AutoAwesome,
            //     inputType = listOf(InOutType.PROMPT),
            //     outputType = listOf(InOutType.IMAGE),
            //     category = AlgorithmCategory.GENERATIVE_AI.displayName,
            //     workflowAsset = "resources/workflow/TextToImage.json",
            //     tags = listOf("generation", "creation", "art"),
            //     parameters = mapOf(
            //         "input_node" to mapOf("OpenCvImageDecode_11" to "path_"),
            //         "output_node" to mapOf("OpenCvImageEncode_26" to "path_"),
            //     ),
            //     processFunction = "processPromptInImageOut"
            // )
        )
    }
    
    /**
     * Filter algorithms by category
     */
    fun getAlgorithmsByCategory(algorithms: List<AIAlgorithm>, category: AlgorithmCategory): List<AIAlgorithm> {
        return algorithms.filter { it.category == category.displayName }
    }
    
    /**
     * Filter algorithms by input type
     */
    fun getAlgorithmsByInputType(algorithms: List<AIAlgorithm>, inputType: InOutType): List<AIAlgorithm> {
        return algorithms.filter { it.inputType.contains(inputType) || it.inputType.contains(InOutType.ALL) }
    }

    /**
     * Get algorithm by name
     */
    fun getAlgorithmsByName(algorithms: List<AIAlgorithm>, name: String): AIAlgorithm? {
        return algorithms.find { it.name == name }
    }

    /**
     * Get algorithm by id
     */
    fun getAlgorithmsById(algorithms: List<AIAlgorithm>, id: String): AIAlgorithm? {
        return algorithms.find { it.id == id }
    } 
    
    /**
     * Search algorithms
     */
    fun searchAlgorithms(algorithms: List<AIAlgorithm>, query: String): List<AIAlgorithm> {
        val lowerQuery = query.lowercase()
        return algorithms.filter { algorithm ->
            algorithm.name.lowercase().contains(lowerQuery) ||
            algorithm.description.lowercase().contains(lowerQuery) ||
            algorithm.tags.any { it.lowercase().contains(lowerQuery) }
        }
    }
}