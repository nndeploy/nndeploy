package com.nndeploy.app

import android.net.Uri
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import coil.compose.AsyncImage
import com.nndeploy.ai.ImageInImageOut
import com.nndeploy.ai.ProcessResult
import com.nndeploy.ai.PromptInPromptOut
import com.nndeploy.ai.PromptInPromptOut.PromptProcessResult
import com.nndeploy.base.*
import kotlinx.coroutines.launch
import android.widget.Toast
import android.util.Log
import androidx.compose.runtime.rememberCoroutineScope
import android.content.Context
import java.io.File
import com.nndeploy.ai.AIAlgorithm
import com.nndeploy.ai.InOutType
import com.nndeploy.ai.AlgorithmFactory

import java.text.SimpleDateFormat
import java.util.Locale
import java.util.Date
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween


/**
 * AI页面ViewModel
 */
class AIViewModel : ViewModel() {
    var selectedAlgorithm by mutableStateOf<AIAlgorithm?>(null)
    var inputUri by mutableStateOf<Uri?>(null)
    var outputUri by mutableStateOf<Uri?>(null)
    var isProcessing by mutableStateOf(false)
    
    // 可用的AI算法列表
    val availableAlgorithms = AlgorithmFactory.createDefaultAlgorithms()
}

/**
 * AI算法首页
 */
@Composable
fun AIScreen(nav: NavHostController, sharedViewModel: AIViewModel = viewModel()) {
    val vm: AIViewModel = sharedViewModel
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF8FAFC))
    ) {
        // 标题栏
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White)
                .padding(16.dp)
        ) {
            Text(
                text = "nndeploy算法中心",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF1E3A8A)
            )
        }
        
        // 算法列表
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // 按类别分组显示
            val groupedAlgorithms = vm.availableAlgorithms.groupBy { it.category }
            
            groupedAlgorithms.forEach { (category, algorithms) ->
                item {
                    Text(
                        text = category,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF374151),
                        modifier = Modifier.padding(vertical = 8.dp)
                    )
                }
                
                items(algorithms) { algorithm ->
                    AIAlgorithmCard(
                        algorithm = algorithm,
                        onClick = {
                            vm.selectedAlgorithm = algorithm
                            nav.navigate("ai_process/${algorithm.id}")
                        }
                    )
                }
            }
        }
    }
}

/**
 * AI算法卡片
 */
@Composable
fun AIAlgorithmCard(
    algorithm: AIAlgorithm,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // 算法图标
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(Color(0xFFE8EEF9)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = algorithm.icon,
                    contentDescription = null,
                    tint = Color(0xFF1E3A8A),
                    modifier = Modifier.size(24.dp)
                )
            }
            
            Spacer(modifier = Modifier.width(16.dp))
            
            // 算法信息
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = algorithm.name,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF111827)
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = algorithm.description,
                    fontSize = 14.sp,
                    color = Color(0xFF6B7280),
                    lineHeight = 20.sp
                )
                Spacer(modifier = Modifier.height(8.dp))
                
                // 支持的输入类型标签
                Row {
                    algorithm.inputType.forEach { type ->
                        InputTypeChip(type)
                        Spacer(modifier = Modifier.width(4.dp))
                    }
                }
            }
            
            // 箭头图标
            Icon(
                imageVector = Icons.Default.ArrowForward,
                contentDescription = null,
                tint = Color(0xFF9CA3AF),
                modifier = Modifier.size(20.dp)
            )
        }
    }
}

/**
 * 输入类型标签
 */
@Composable
fun InputTypeChip(inputType: InOutType) {
    val (text, color) = when (inputType) {
        InOutType.IMAGE -> "图片" to Color(0xFF10B981)
        InOutType.VIDEO -> "视频" to Color(0xFFF59E0B)
        InOutType.CAMERA -> "摄像头" to Color(0xFFEF4444)
        InOutType.AUDIO -> "音频" to Color(0xFFFF6B6B)
        InOutType.TEXT -> "文本" to Color(0xFF4ECDC4)
        InOutType.PROMPT -> "提示词" to Color(0xFF8B5CF6)
        InOutType.ALL -> "全支持" to Color(0xFF06B6D4)
    }
    
    Box(
        modifier = Modifier
            .background(
                color = color.copy(alpha = 0.1f),
                shape = RoundedCornerShape(12.dp)
            )
            .padding(horizontal = 8.dp, vertical = 4.dp)
    ) {
        Text(
            text = text,
            fontSize = 12.sp,
            color = color,
            fontWeight = FontWeight.Medium
        )
    }
}

/**
 * AI算法处理页面
 */
@Composable
fun CVProcessScreen(
    nav: NavHostController,
    algorithmId: String,
    sharedViewModel: AIViewModel = viewModel()
) {
    val vm: AIViewModel = sharedViewModel
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    // 找到对应的算法
    val algorithm = AlgorithmFactory.getAlgorithmsById(vm.availableAlgorithms, algorithmId)
    
    // 图片选择器
    val imagePickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        vm.inputUri = uri
    }
    
    // 视频选择器
    val videoPickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        vm.inputUri = uri
    }
    
    // 拍照
    val cameraLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            // 这里需要处理拍照成功的情况
            Log.w("CVProcessScreen", "Photo taken successfully")
        }
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF8FAFC))
    ) {
        // 顶部栏
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White)
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = { nav.popBackStack() }) {
                Icon(
                    imageVector = Icons.Default.ArrowBack,
                    contentDescription = "返回"
                )
            }
            Text(
                text = algorithm?.name ?: "AI处理",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF1E3A8A),
                modifier = Modifier.weight(1f)
            )
        }
        
        // 输入区域
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(16.dp),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                if (vm.inputUri != null) {
                    AsyncImage(
                        model = vm.inputUri,
                        contentDescription = "输入内容",
                        modifier = Modifier.fillMaxWidth(0.8f)
                    )
                } else {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Icon(
                            imageVector = Icons.Default.CloudUpload,
                            contentDescription = null,
                            tint = Color(0xFF9CA3AF),
                            modifier = Modifier.size(48.dp)
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = "选择输入内容",
                            fontSize = 16.sp,
                            color = Color(0xFF6B7280)
                        )
                    }
                }
            }
        }
        
        // 输入选择按钮
        algorithm?.let { algo ->
            InputSelectionButtons(
                inputTypes = algo.inputType,
                onImageSelect = { imagePickerLauncher.launch("image/*") },
                onVideoSelect = { videoPickerLauncher.launch("video/*") },
                onCameraPhoto = { 
                    // 创建临时文件用于拍照
                    val photoUri = CameraUtils.createPhotoUri(context)
                    vm.inputUri = photoUri
                    cameraLauncher.launch(photoUri)
                },
                onCameraVideo = {
                    // 录像功能实现
                    Toast.makeText(context, "录像功能待实现", Toast.LENGTH_SHORT).show()
                }
            )
        }
        
        // 处理按钮
        Button(
            onClick = {
                vm.inputUri?.let { uri ->
                    scope.launch {
                        vm.isProcessing = true
                        // 判断algorithm是否存在
                        if (algorithm == null) {
                            Toast.makeText(context, "算法 $algorithmId 不存在", Toast.LENGTH_LONG).show()
                            return@launch
                        }
                        try {
                            val result = ImageInImageOut.processImageInImageOut(context, uri, algorithm!!)
                            
                            when (result) {
                                is ProcessResult.Success -> {
                                    vm.outputUri = result.resultUri
                                    nav.navigate("ai_result")
                                }
                                is ProcessResult.Error -> {
                                    Toast.makeText(context, result.message, Toast.LENGTH_LONG).show()
                                }
                            }
                        } finally {
                            vm.isProcessing = false
                        }
                    }
                } ?: Toast.makeText(context, "请先选择输入内容", Toast.LENGTH_SHORT).show()
            },
            enabled = vm.inputUri != null && !vm.isProcessing,
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
                .height(56.dp),
            shape = RoundedCornerShape(28.dp)
        ) {
            if (vm.isProcessing) {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp),
                    color = Color.White
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("处理中...")
            } else {
                Text("开始处理", fontSize = 16.sp)
            }
        }
    }
}

/**
 * 输入选择按钮组
 */
@Composable
fun InputSelectionButtons(
    inputTypes: List<InOutType>,
    onImageSelect: () -> Unit,
    onVideoSelect: () -> Unit,
    onCameraPhoto: () -> Unit,
    onCameraVideo: () -> Unit
) {
    Column(
        modifier = Modifier.padding(16.dp)
    ) {
        Text(
            text = "选择输入方式",
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF374151),
            modifier = Modifier.padding(bottom = 12.dp)
        )
        
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // 根据算法支持的输入类型显示对应按钮
            if (inputTypes.contains(InOutType.IMAGE)) {
                OutlinedButton(
                    onClick = onImageSelect,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.Photo, null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("相册")
                }
                OutlinedButton(
                    onClick = onCameraPhoto,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.CameraAlt, null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("拍照")
                }
            }
        }
    }
}

/**
 * AI处理结果页面
 */
@Composable
fun CVResultScreen(nav: NavHostController, sharedViewModel: AIViewModel = viewModel()) {
    val vm: AIViewModel = sharedViewModel
    val context = LocalContext.current
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF8FAFC))
    ) {
        // 顶部栏
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White)
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = { nav.popBackStack() }) {
                Icon(
                    imageVector = Icons.Default.ArrowBack,
                    contentDescription = "返回"
                )
            }
            Text(
                text = "处理结果",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF1E3A8A),
                modifier = Modifier.weight(1f)
            )
        }
        
        // 结果展示区域
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(16.dp),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                if (vm.outputUri != null) {
                    AsyncImage(
                        model = vm.outputUri,
                        contentDescription = "处理结果",
                        modifier = Modifier.fillMaxWidth(0.9f)
                    )
                } else {
                    Text(
                        text = "暂无处理结果",
                        fontSize = 16.sp,
                        color = Color(0xFF6B7280)
                    )
                }
            }
        }
        
        // 操作按钮
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            OutlinedButton(
                onClick = {
                    vm.outputUri?.let { uri ->
                        if (FileUtils.saveCopyToDownloads(context, uri)) {
                            Toast.makeText(context, "保存成功", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(context, "保存失败", Toast.LENGTH_SHORT).show()
                        }
                    }
                },
                modifier = Modifier.weight(1f)
            ) {
                Icon(Icons.Default.Download, null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(4.dp))
                Text("保存")
            }
            
            Button(
                onClick = {
                    vm.outputUri?.let { uri ->
                        val shareIntent = android.content.Intent().apply {
                            action = android.content.Intent.ACTION_SEND
                            type = "image/*"
                            putExtra(android.content.Intent.EXTRA_STREAM, uri)
                            addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }
                        context.startActivity(android.content.Intent.createChooser(shareIntent, "分享结果"))
                    }
                },
                modifier = Modifier.weight(1f)
            ) {
                Icon(Icons.Default.Share, null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(4.dp))
                Text("分享")
            }
        }
        
        // 继续处理按钮
        Button(
            onClick = { nav.navigate("ai") },
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF10B981)),
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp)
                .height(52.dp),
            shape = RoundedCornerShape(26.dp)
        ) {
            Text("继续处理其他算法", fontSize = 16.sp)
        }
    }
}

/**
 * 根据URI确定输入媒体类型
 */
private fun determineInputType(uri: Uri, context: Context): com.nndeploy.ai.InputMediaType {
    return try {
        val mimeType = context.contentResolver.getType(uri)
        when {
            mimeType?.startsWith("image/") == true -> com.nndeploy.ai.InputMediaType.IMAGE
            mimeType?.startsWith("video/") == true -> com.nndeploy.ai.InputMediaType.VIDEO
            else -> com.nndeploy.ai.InputMediaType.IMAGE // 默认为图片
        }
    } catch (e: Exception) {
        com.nndeploy.ai.InputMediaType.IMAGE // 异常时默认为图片
    }
}

/**
 * LLM聊天处理页面
 */
@Composable
fun LlmChatProcessScreen(
    nav: NavHostController,
    algorithmId: String,
    sharedViewModel: AIViewModel = viewModel()
) {
    val vm: AIViewModel = sharedViewModel
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    // 找到对应的算法
    val algorithm = AlgorithmFactory.getAlgorithmsById(vm.availableAlgorithms, algorithmId)
    
    // 聊天消息状态
    var messages by remember { mutableStateOf(listOf<ChatMessage>()) }
    var inputText by remember { mutableStateOf("") }
    var isTyping by remember { mutableStateOf(false) }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF8FAFC))
    ) {
        // 顶部栏
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White)
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = { nav.popBackStack() }) {
                Icon(
                    imageVector = Icons.Default.ArrowBack,
                    contentDescription = "返回"
                )
            }
            Text(
                text = algorithm?.name ?: "AI聊天",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF1E3A8A),
                modifier = Modifier.weight(1f)
            )
            IconButton(
                onClick = { 
                    messages = listOf()
                }
            ) {
                Icon(
                    imageVector = Icons.Default.Refresh,
                    contentDescription = "清空聊天"
                )
            }
        }
        
        // 聊天消息区域
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
            contentPadding = PaddingValues(vertical = 16.dp)
        ) {
            if (messages.isEmpty()) {
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(containerColor = Color.White)
                    ) {
                        Column(
                            modifier = Modifier.padding(24.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Icon(
                                imageVector = Icons.Default.Chat,
                                contentDescription = null,
                                tint = Color(0xFF9CA3AF),
                                modifier = Modifier.size(48.dp)
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = "开始与AI对话",
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color(0xFF374151)
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "输入您的问题，AI将为您提供帮助",
                                fontSize = 14.sp,
                                color = Color(0xFF6B7280),
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                }
            } else {
                items(messages) { message ->
                    ChatMessageItem(message = message)
                }
            }
            
            // 正在输入指示器
            if (isTyping) {
                item {
                    TypingIndicator()
                }
            }
        }
        
        // 输入区域
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            shape = RoundedCornerShape(28.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp),
                verticalAlignment = Alignment.Bottom
            ) {
                OutlinedTextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    placeholder = { Text("输入您的问题...") },
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 8.dp),
                    shape = RoundedCornerShape(20.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Color(0xFF10B981),
                        unfocusedBorderColor = Color(0xFFE5E7EB)
                    ),
                    maxLines = 4
                )
                
                IconButton(
                    onClick = {
                        if (inputText.isNotBlank() && !isTyping) {
                            val userMessage = ChatMessage(
                                content = inputText,
                                isUser = true,
                                timestamp = System.currentTimeMillis()
                            )
                            messages = messages + userMessage
                            
                            scope.launch {
                                isTyping = true
                                try {
                                    // 判断algorithm是否存在
                                    if (algorithm == null) {
                                        Toast.makeText(context, "算法 $algorithmId 不存在", Toast.LENGTH_LONG).show()
                                        return@launch
                                    }
                                    
                                    Log.d("LlmChatProcessScreen", "inputText: $inputText")
                                    val result = PromptInPromptOut.processPromptInPromptOut(context, inputText, algorithm)
                                    
                                    when (result) {
                                        is PromptProcessResult.Success -> {
                                            val aiMessage = ChatMessage(
                                                content = result.response,
                                                isUser = false,
                                                timestamp = System.currentTimeMillis()
                                            )
                                            messages = messages + aiMessage
                                        }
                                        is PromptProcessResult.Error -> {
                                            val errorMessage = ChatMessage(
                                                content = "抱歉，处理出现错误：${result.message}",
                                                isUser = false,
                                                timestamp = System.currentTimeMillis(),
                                                isError = true
                                            )
                                            messages = messages + errorMessage
                                        }
                                    }
                                } catch (e: Exception) {
                                    val errorMessage = ChatMessage(
                                        content = "抱歉，发生了未知错误",
                                        isUser = false,
                                        timestamp = System.currentTimeMillis(),
                                        isError = true
                                    )
                                    messages = messages + errorMessage
                                } finally {
                                    isTyping = false
                                }
                            }
                            
                            inputText = ""
                        }
                    },
                    enabled = inputText.isNotBlank() && !isTyping,
                    modifier = Modifier
                        .size(48.dp)
                        .background(
                            if (inputText.isNotBlank() && !isTyping) Color(0xFF10B981) else Color(0xFFE5E7EB),
                            CircleShape
                        )
                ) {
                    Icon(
                        imageVector = Icons.Default.Send,
                        contentDescription = "发送",
                        tint = if (inputText.isNotBlank() && !isTyping) Color.White else Color(0xFF9CA3AF)
                    )
                }
            }
        }
    }
}

/**
 * 聊天消息数据类
 */
data class ChatMessage(
    val content: String,
    val isUser: Boolean,
    val timestamp: Long,
    val isError: Boolean = false
)

/**
 * 聊天消息项组件
 */
@Composable
fun ChatMessageItem(message: ChatMessage) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (message.isUser) Arrangement.End else Arrangement.Start
    ) {
        if (message.isUser) {
            Spacer(modifier = Modifier.width(48.dp))
        }
        
        Card(
            modifier = Modifier.widthIn(max = 280.dp),
            shape = RoundedCornerShape(
                topStart = 16.dp,
                topEnd = 16.dp,
                bottomStart = if (message.isUser) 16.dp else 4.dp,
                bottomEnd = if (message.isUser) 4.dp else 16.dp
            ),
            colors = CardDefaults.cardColors(
                containerColor = when {
                    message.isError -> Color(0xFFFEE2E2)
                    message.isUser -> Color(0xFF10B981)
                    else -> Color.White
                }
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(
                modifier = Modifier.padding(12.dp)
            ) {
                Text(
                    text = message.content,
                    fontSize = 14.sp,
                    color = when {
                        message.isError -> Color(0xFFDC2626)
                        message.isUser -> Color.White
                        else -> Color(0xFF374151)
                    },
                    lineHeight = 20.sp
                )
                
                Spacer(modifier = Modifier.height(4.dp))
                
                Text(
                    text = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(message.timestamp)),
                    fontSize = 10.sp,
                    color = when {
                        message.isError -> Color(0xFFDC2626).copy(alpha = 0.7f)
                        message.isUser -> Color.White.copy(alpha = 0.7f)
                        else -> Color(0xFF9CA3AF)
                    }
                )
            }
        }
        
        if (!message.isUser) {
            Spacer(modifier = Modifier.width(48.dp))
        }
    }
}

/**
 * 正在输入指示器
 */
@Composable
fun TypingIndicator() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Start
    ) {
        Card(
            modifier = Modifier.widthIn(max = 100.dp),
            shape = RoundedCornerShape(
                topStart = 16.dp,
                topEnd = 16.dp,
                bottomStart = 4.dp,
                bottomEnd = 16.dp
            ),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Row(
                modifier = Modifier.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                repeat(3) { index ->
                    val animatedAlpha by animateFloatAsState(
                        targetValue = if ((System.currentTimeMillis() / 500) % 3 == index.toLong()) 1f else 0.3f,
                        animationSpec = tween(500),
                        label = "typing_dot_$index"
                    )
                    
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .background(
                                Color(0xFF9CA3AF).copy(alpha = animatedAlpha),
                                CircleShape
                            )
                    )
                    
                    if (index < 2) {
                        Spacer(modifier = Modifier.width(4.dp))
                    }
                }
            }
        }
        
        Spacer(modifier = Modifier.width(48.dp))
    }
}

