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
import com.nndeploy.base.*
import kotlinx.coroutines.launch
import android.widget.Toast
import android.util.Log
import androidx.compose.runtime.rememberCoroutineScope
import android.content.Context
import java.io.File


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
    val workflowAsset: String
)

/**
 * 输入类型枚举
 */
enum class InOutType {
    IMAGE,      // 仅图片
    VIDEO,      // 仅视频  
    CAMERA,     // 仅摄像头
    PROMPT,          // 提示词
    ALL              // 全部支持
}

/**
 * 输入源类型
 */
enum class InputSource {
    GALLERY_IMAGE,   // 相册图片
    GALLERY_VIDEO,   // 相册视频
    CAMERA_PHOTO,    // 拍照
    CAMERA_VIDEO,    // 录像
    TEXT_INPUT       // 文本框
}

/**
 * AI页面ViewModel
 */
class AIViewModel : ViewModel() {
    var selectedAlgorithm by mutableStateOf<AIAlgorithm?>(null)
    var inputUri by mutableStateOf<Uri?>(null)
    var outputUri by mutableStateOf<Uri?>(null)
    var isProcessing by mutableStateOf(false)
    
    // 可用的AI算法列表
    val availableAlgorithms = listOf(
        AIAlgorithm(
            id = "image_segmentation",
            name = "图像分割",
            description = "智能识别并分割图像中的不同对象和区域",
            icon = Icons.Default.Crop,
            inputType = listOf(InOutType.IMAGE),
            outputType = listOf(InOutType.IMAGE),
            category = "计算机视觉",
            workflowAsset = "image_segmentation.json"
        )
    )
}

/**
 * AI算法首页
 */
@Composable
fun AIScreen(nav: NavHostController) {
    val vm: AIViewModel = viewModel()
    
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
                text = "AI算法中心",
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
fun AIProcessScreen(
    nav: NavHostController,
    algorithmId: String
) {
    val vm: AIViewModel = viewModel()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    // 找到对应的算法
    val algorithm = vm.availableAlgorithms.find { it.id == algorithmId }
    
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
            Log.w("AIProcessScreen", "Photo taken successfully")
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
                        try {
                            val inputType = determineInputType(uri, context)
                            val result = when (algorithmId) {
                                "image_segmentation" -> {
                                    ImageInImageOut.processImageInImageOut(context, uri, inputType)
                                }
                                else -> {
                                    ProcessResult.Error("算法 $algorithmId 暂未实现")
                                }
                            }
                            
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
fun AIResultScreen(nav: NavHostController) {
    val vm: AIViewModel = viewModel()
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
