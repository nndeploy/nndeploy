package com.nndeploy.ai

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.ui.graphics.vector.ImageVector

/**
 * AI算法信息数据类
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
    val parameters: Map<String, Any> = emptyMap(), // 算法参数
    val processFunction: String = "none"
)

/**
 * 输入输出类型枚举
 */
enum class InOutType(val displayName: String, val icon: ImageVector) {
    IMAGE("图片", Icons.Default.Image),
    VIDEO("视频", Icons.Default.VideoFile),
    CAMERA("摄像头", Icons.Default.CameraAlt),
    PROMPT("提示词", Icons.Default.TextFields),
    AUDIO("音频", Icons.Default.AudioFile),
    TEXT("文本", Icons.Default.Description),
    ALL("全支持", Icons.Default.AllInclusive);
    
    companion object {
        fun fromString(type: String): InOutType? {
            return values().find { it.name.equals(type, ignoreCase = true) }
        }
    }
}

/**
 * 输入媒体类型枚举
 */
enum class InputMediaType {
    IMAGE,
    VIDEO,
    CAMERA,
    AUDIO,
    TEXT
}

/**
 * 算法类别枚举
 */
enum class AlgorithmCategory(val displayName: String, val icon: ImageVector) {
    COMPUTER_VISION("计算机视觉", Icons.Default.Visibility),
    NATURAL_LANGUAGE("自然语言处理", Icons.Default.TextFields),
    AUDIO_PROCESSING("音频处理", Icons.Default.AudioFile),
    GENERATIVE_AI("生成式AI", Icons.Default.AutoAwesome),
    OTHER("其他", Icons.Default.Category)
}

/**
 * 算法处理状态
 */
enum class ProcessingStatus {
    IDLE,           // 空闲
    LOADING,        // 加载中
    PROCESSING,     // 处理中
    COMPLETED,      // 完成
    ERROR,          // 错误
    CANCELLED       // 已取消
}

/**
 * 算法运行时信息
 */
data class AlgorithmRuntime(
    val status: ProcessingStatus = ProcessingStatus.IDLE,
    val progress: Float = 0f,           // 处理进度 0.0-1.0
    val startTime: Long = 0L,           // 开始时间
    val endTime: Long = 0L,             // 结束时间
    val errorMessage: String? = null
)

/**
 * 算法工厂类
 */
object AlgorithmFactory {
    
    /**
     * 创建预定义的算法列表
     */
    fun createDefaultAlgorithms(): List<AIAlgorithm> {
        return listOf(
            // 计算机视觉算法
            AIAlgorithm(
                id = "image_segmentation",
                name = "图像分割",
                description = "智能识别并分割图像中的不同对象和区域",
                icon = Icons.Default.Crop,
                inputType = listOf(InOutType.IMAGE),
                outputType = listOf(InOutType.IMAGE),
                category = AlgorithmCategory.COMPUTER_VISION.displayName,
                workflowAsset = "resources/workflow/SegmentRMBGMNN.json",
                tags = listOf("分割", "目标检测", "图像处理"),
                parameters = mapOf(
                    "input_node" to mapOf("OpenCvImageDecode_11" to "path_"),
                    "output_node" to mapOf("OpenCvImageEncode_16" to "path_"),
                ),
                processFunction = "processImageInImageOut"
            ),
            AIAlgorithm(
                id = "image_classification",
                name = "图像分类",
                description = "智能识别图像中的物体类别和标签",
                icon = Icons.Default.Category,
                inputType = listOf(InOutType.IMAGE),
                outputType = listOf(InOutType.TEXT),
                category = AlgorithmCategory.COMPUTER_VISION.displayName,
                workflowAsset = "resources/workflow/ClassificationResNetMnn.json",
                tags = listOf("分类", "识别", "标签"),
                parameters = mapOf(
                    "input_node" to mapOf("OpenCvImageDecode_11" to "path_"),
                    "output_node" to mapOf("OpenCvImageEncode_26" to "path_"),
                ),
                processFunction = "processImageInImageOut"
            ),
            
            // 自然语言处理算法
            AIAlgorithm(
                id = "text_chat",
                name = "LLM聊天",
                description = "基于大语言模型的智能对话系统，支持多轮对话和上下文理解",
                icon = Icons.Default.QuestionAnswer,
                inputType = listOf(InOutType.PROMPT),
                outputType = listOf(InOutType.TEXT),
                category = AlgorithmCategory.NATURAL_LANGUAGE.displayName,
                workflowAsset = "resources/workflow/QwenMNN.json",
                tags = listOf("对话", "聊天", "问答"),
                parameters = mapOf(
                    "input_node" to mapOf("Prompt_4" to "user_content_"),
                    "output_node" to mapOf("LlmOut_3" to "path_"),
                ),
                processFunction = "processPromptInPromptOut"
            ),
            
            // 生成式AI算法
            AIAlgorithm(
                id = "text_to_image",
                name = "文本生成图像",
                description = "根据文本描述生成对应的图像内容，支持多种艺术风格",
                icon = Icons.Default.AutoAwesome,
                inputType = listOf(InOutType.PROMPT),
                outputType = listOf(InOutType.IMAGE),
                category = AlgorithmCategory.GENERATIVE_AI.displayName,
                workflowAsset = "resources/workflow/TextToImage.json",
                tags = listOf("生成", "创作", "艺术"),
                parameters = mapOf(
                    "input_node" to mapOf("OpenCvImageDecode_11" to "path_"),
                    "output_node" to mapOf("OpenCvImageEncode_26" to "path_"),
                ),
                processFunction = "processPromptInImageOut"
            )
        )
    }
    
    /**
     * 根据类别筛选算法
     */
    fun getAlgorithmsByCategory(algorithms: List<AIAlgorithm>, category: AlgorithmCategory): List<AIAlgorithm> {
        return algorithms.filter { it.category == category.displayName }
    }
    
    /**
     * 根据输入类型筛选算法
     */
    fun getAlgorithmsByInputType(algorithms: List<AIAlgorithm>, inputType: InOutType): List<AIAlgorithm> {
        return algorithms.filter { it.inputType.contains(inputType) || it.inputType.contains(InOutType.ALL) }
    }

    /**
     * 根据名字返回算法
     */
    fun getAlgorithmsByName(algorithms: List<AIAlgorithm>, name: String): AIAlgorithm? {
        return algorithms.find { it.name == name }
    }

    /**
     * 根据id返回算法
     */
    fun getAlgorithmsById(algorithms: List<AIAlgorithm>, id: String): AIAlgorithm? {
        return algorithms.find { it.id == id }
    } 
    
    /**
     * 搜索算法
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