package com.example.imageapp

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.History
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import coil.compose.AsyncImage
import com.example.imageapp.ui.theme.AppTheme
import com.nndeploy.dag.GraphRunner
import kotlinx.coroutines.launch
import androidx.compose.runtime.rememberCoroutineScope
import android.widget.Toast
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.w("MainActivity", "onCreate called")
        setContent {
            AppTheme { App() }
        }
    }
}

class AppVM: ViewModel() {
    var pickedImage by mutableStateOf<Uri?>(null)
    var processedImage by mutableStateOf<Uri?>(null)
}

@Composable
fun App() {
    val nav = rememberNavController()
    val vm: AppVM = viewModel()
    Log.w("App", "App composable initialized")
    Scaffold(
        bottomBar = { BottomBar(nav) }
    ) { inner ->
        NavHost(
            navController = nav,
            startDestination = "home",
            modifier = Modifier.padding(inner)
        ) {
            composable("home") { UploadScreen(nav, vm) }
            composable("process") { ProcessScreen(nav, vm) }
            composable("result") { ResultScreen(nav, vm) }
            composable("history") { HistoryScreen() }
        }
    }
}

@Composable
fun BottomBar(nav: NavHostController) {
    NavigationBar {
        NavigationBarItem(
            selected = false, onClick = { 
                Log.w("BottomBar", "Navigate to home")
                nav.navigate("home") 
            },
            icon = { Icon(Icons.Default.Description, contentDescription = "首页") }, label = { Text("首页") }
        )
        NavigationBarItem(
            selected = false, onClick = { 
                Log.w("BottomBar", "Navigate to process")
                nav.navigate("process") 
            },
            icon = { Icon(Icons.Default.History, contentDescription = "处理") }, label = { Text("处理") }
        )
        NavigationBarItem(
            selected = false, onClick = { 
                Log.w("BottomBar", "Navigate to history")
                nav.navigate("history") 
            },
            icon = { Icon(Icons.Default.History, contentDescription = "历史") }, label = { Text("历史") }
        )
    }
}

@Composable
fun UploadScreen(nav: NavHostController, vm: AppVM) {
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        Log.w("UploadScreen", "Image picked: $uri")
        vm.pickedImage = uri
        if (uri != null) {
            Log.w("UploadScreen", "Navigating to process screen")
            nav.navigate("process")
        }
    }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.height(24.dp))
        Text("图片上传", fontSize = 36.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1E3A8A))
        Spacer(Modifier.height(8.dp))
        Text("选择或拍摄图片开始处理", color = Color.Gray)
        Spacer(Modifier.height(24.dp))
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            shape = RoundedCornerShape(24.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 6.dp)
        ) {
            Column(
                Modifier.fillMaxSize(),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Box(
                    modifier = Modifier
                        .size(80.dp)
                        .clip(CircleShape)
                        .background(Color(0xFFE8EEF9)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.CloudUpload, contentDescription = null, tint = Color(0xFF1E3A8A), modifier = Modifier.size(40.dp))
                }
                Spacer(Modifier.height(16.dp))
                Text("上传您的图片", fontSize = 24.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(8.dp))
                Text("支持JPG/PNG格式，最大10MB", color = Color.Gray)
                Spacer(Modifier.height(24.dp))
                Button(onClick = { 
                    Log.w("UploadScreen", "Select image button clicked")
                    launcher.launch("image/*") 
                }) {
                    Text("选择图片")
                }
            }
        }
    }
}

@Composable
fun ProcessScreen(nav: NavHostController, vm: AppVM) {
    val uri = vm.pickedImage
    val context = LocalContext.current
    var selected by remember { mutableStateOf("分割") }
    val scope = rememberCoroutineScope()
    
    Log.w("ProcessScreen", "ProcessScreen loaded with image: $uri")
    
    Column(Modifier.fillMaxSize()) {
        Box(Modifier.fillMaxWidth().weight(1f).padding(16.dp)) {
            Card(
                modifier = Modifier.fillMaxSize(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFF1F5F9))
            ) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    if (uri != null) {
                        AsyncImage(model = uri, contentDescription = null, modifier = Modifier.fillMaxWidth(0.7f))
                    } else {
                        Text("未选择图片")
                    }
                }
            }
        }
        Text("处理方式", modifier = Modifier.padding(16.dp), fontSize = 22.sp, fontWeight = FontWeight.Bold)
        Row(Modifier.padding(horizontal = 16.dp)) {
            FilterChip(label = "分割", selected = selected == "分割") { 
                Log.w("ProcessScreen", "Selected processing type: 分割")
                selected = "分割" 
            }
            Spacer(Modifier.width(12.dp))
            FilterChip(label = "抠脸", selected = selected == "抠脸") { 
                Log.w("ProcessScreen", "Selected processing type: 抠脸")
                selected = "抠脸" 
            }
            Spacer(Modifier.width(12.dp))
            FilterChip(label = "灰度图", selected = selected == "灰度图") { 
                Log.w("ProcessScreen", "Selected processing type: 灰度图")
                selected = "灰度图" 
            }
        }
        Spacer(Modifier.height(12.dp))
        Button(
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(28.dp),
            onClick = {
                Log.w("ProcessScreen", "Start processing button clicked")
                scope.launch {
                    try {
                        if (selected == "灰度图" && uri != null) {
                            Log.w("ProcessScreen", "Starting grayscale conversion")
                            val processedUri = convertToGrayscale(context, uri)
                            if (processedUri != null) {
                                vm.processedImage = processedUri
                                Log.w("ProcessScreen", "Grayscale conversion successful")
                                Toast.makeText(context, "灰度图转换成功", Toast.LENGTH_SHORT).show()
                            } else {
                                Log.e("ProcessScreen", "Grayscale conversion failed")
                                Toast.makeText(context, "灰度图转换失败", Toast.LENGTH_SHORT).show()
                                vm.processedImage = uri
                            }
                        } else {
                            // 内部私有目录
                            // Log.w("ProcessScreen", "Initializing GraphRunner")
                            // // 1) 确保资源就绪
                            // ensureResourcesReady(context)
                            // val baseDir = File(context.filesDir, "resources").absolutePath

                            // // 2) 读取 assets 中的 workflow，并替换占位符
                            // val workflowAsset = "resources/workflow/ClassificationResNetMnn.json" // 或你的实际路径
                            // val rawJson = context.assets.open(workflowAsset).bufferedReader().use { it.readText() }
                            // val resolvedJson = rawJson.replace("resources/", "$baseDir/".replace("\\", "/"))
                            // Log.w("ProcessScreen", "Resolved JSON: $resolvedJson")
                                
                            // // 3) 写入 filesDir，得到真实文件路径
                            // val workflowOut = File(context.filesDir, "workflow/ClassificationResNetMnn_resolved.json")
                            // workflowOut.parentFile?.mkdirs()
                            // workflowOut.writeText(resolvedJson)

                            // // 4) 调用底层（以文件路径）
                            // val runner = GraphRunner()
                            // runner.setJsonFile(true)
                            // runner.setTimeProfile(true)
                            // runner.setDebug(true)
                            // // 可选：把基准目录传下去，便于底层解析相对路径
                            // // runner.setNodeValue("__global__", "base_dir", baseDir)

                            // val ok = runner.run(workflowOut.absolutePath, "seg_demo", "task_${System.currentTimeMillis()}")
                            // runner.close()
                            // vm.processedImage = uri

                            // 1) 确保外部资源就绪
                            val extResDir = ensureExternalResourcesReady(context)
                            val extRoot = getExternalRoot(context)
                            val extWorkflowDir = java.io.File(extRoot, "workflow").apply { mkdirs() }

                            // 2) 读取 assets 的 workflow，并把相对路径替换为外部绝对路径
                            val workflowAsset = "resources/workflow/ClassificationResNetMnn.json"
                            val rawJson = context.assets.open(workflowAsset).bufferedReader().use { it.readText() }
                            val resolvedJson = rawJson.replace("resources/", "${extResDir.absolutePath}/".replace("\\", "/"))
                            Log.w("ProcessScreen", "Resolved JSON: $resolvedJson")

                            // 3) 写到外部私有目录，得到真实文件路径
                            val workflowOut = java.io.File(extWorkflowDir, "ClassificationResNetMnn_resolved.json").apply {
                                writeText(resolvedJson)
                            }

                            // 4) 以文件路径运行底层
                            val runner = GraphRunner()
                            runner.setJsonFile(true)
                            runner.setTimeProfile(true)
                            runner.setDebug(true)
                            // runner.setNodeValue("__global__", "base_dir", extResDir.absolutePath) // 如需下发基准目录
                            val ok = runner.run(workflowOut.absolutePath, "seg_demo", "task_${System.currentTimeMillis()}")
                            runner.close()
                            
                            
                            // 获取结果图片路径
                            val resultImagePath = java.io.File(extResDir, "images/result.resnet.jpg")
                            if (resultImagePath.exists()) {
                                vm.processedImage = Uri.fromFile(resultImagePath)
                            } else {
                                Log.w("ProcessScreen", "Result image not found at: ${resultImagePath.absolutePath}")
                                vm.processedImage = uri // 使用原图作为备选
                            }
                        }
                    } catch (e: Throwable) {
                        Log.e("ProcessScreen", "JNI exception occurred", e)
                        Toast.makeText(context, "JNI 异常: ${e.message}", Toast.LENGTH_SHORT).show()
                        vm.processedImage = uri
                    } finally {
                        Log.w("ProcessScreen", "Navigating to result screen")
                        nav.navigate("result")
                    }
                }
            }
        ) {
            Text("开始处理", fontSize = 18.sp)
        }
    }
}

@Composable
private fun FilterChip(label: String, selected: Boolean, onClick: () -> Unit) {
    AssistChip(
        onClick = onClick,
        label = { Text(label) },
        shape = RoundedCornerShape(16.dp),
        colors = AssistChipDefaults.assistChipColors(
            containerColor = if (selected) Color(0xFFDCE7FF) else Color(0xFFF8FAFC)
        )
    )
}

@Composable
fun ResultScreen(nav: NavHostController, vm: AppVM) {
    val context = LocalContext.current
    val resultUri = vm.processedImage
    
    Log.w("ResultScreen", "ResultScreen loaded with image: $resultUri")
    
    Column(Modifier.fillMaxSize()) {
        Box(Modifier.fillMaxWidth().weight(1f).padding(16.dp)) {
            Card(
                modifier = Modifier.fillMaxSize(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFC))
            ) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    if (resultUri != null) {
                        AsyncImage(model = resultUri, contentDescription = null, modifier = Modifier.fillMaxWidth(0.8f))
                    } else {
                        Text(
                            text = "图片加载失败",
                            color = Color.DarkGray,
                            fontSize = 20.sp,
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }
        }
        Row(Modifier.padding(16.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(
                onClick = {
                    Log.w("ResultScreen", "Download button clicked")
                    resultUri?.let { 
                        Log.w("ResultScreen", "Saving image to downloads: $it")
                        saveCopyToDownloads(context, it) 
                    }
                },
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF183D8C)),
                modifier = Modifier.weight(1f).height(52.dp)
            ) { Text("下载图片") }
            Button(
                onClick = {
                    Log.w("ResultScreen", "Share button clicked")
                    // Share
                    resultUri?.let {
                        Log.w("ResultScreen", "Sharing image: $it")
                        val share = android.content.Intent().apply {
                            action = android.content.Intent.ACTION_SEND
                            type = "image/*"
                            putExtra(android.content.Intent.EXTRA_STREAM, it)
                            addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }
                        context.startActivity(android.content.Intent.createChooser(share, "分享图片"))
                    }
                },
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFF59E0B)),
                modifier = Modifier.weight(1f).height(52.dp)
            ) { Text("分享图片") }
        }
        Button(
            onClick = { 
                Log.w("ResultScreen", "Continue processing button clicked")
                nav.navigate("process") 
            },
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF10B981)),
            modifier = Modifier
                .padding(horizontal = 16.dp, vertical = 8.dp)
                .fillMaxWidth()
                .height(52.dp)
        ) { Text("继续处理") }
        Spacer(Modifier.height(8.dp))
    }
}

@Composable
fun HistoryScreen() {
    Log.w("HistoryScreen", "HistoryScreen loaded")
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("历史（占位）")
    }
}

private fun convertToGrayscale(context: android.content.Context, uri: Uri): Uri? {
    return try {
        Log.w("convertToGrayscale", "Starting grayscale conversion for URI: $uri")
        
        // 读取原始图片
        val resolver = context.contentResolver
        val inputStream = resolver.openInputStream(uri)
        val originalBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(resolver, uri)
            ImageDecoder.decodeBitmap(source)
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(resolver, uri)
        }
        inputStream?.close()
        
        Log.w("convertToGrayscale", "Original bitmap size: ${originalBitmap.width}x${originalBitmap.height}")
        
        // 创建灰度图bitmap
        val width = originalBitmap.width
        val height = originalBitmap.height
        val grayscaleBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        
        // 逐像素转换为灰度
        for (x in 0 until width) {
            for (y in 0 until height) {
                val pixel = originalBitmap.getPixel(x, y)
                
                // 提取RGB分量
                val red = (pixel shr 16) and 0xFF
                val green = (pixel shr 8) and 0xFF
                val blue = pixel and 0xFF
                
                // 使用加权平均法计算灰度值 (0.299*R + 0.587*G + 0.114*B)
                val gray = (0.299 * red + 0.587 * green + 0.114 * blue).toInt()
                
                // 设置灰度像素 (保持alpha通道)
                val alpha = (pixel shr 24) and 0xFF
                val grayPixel = (alpha shl 24) or (gray shl 16) or (gray shl 8) or gray
                grayscaleBitmap.setPixel(x, y, grayPixel)
            }
        }
        
        Log.w("convertToGrayscale", "Grayscale conversion completed")
        
        // 保存灰度图到临时文件
        val tempFile = File(context.cacheDir, "grayscale_${System.currentTimeMillis()}.jpg")
        val outputStream = FileOutputStream(tempFile)
        grayscaleBitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
        outputStream.close()
        
        // 释放bitmap资源
        originalBitmap.recycle()
        grayscaleBitmap.recycle()
        
        Log.w("convertToGrayscale", "Grayscale image saved to: ${tempFile.absolutePath}")
        
        // 返回文件URI
        android.net.Uri.fromFile(tempFile)
        
    } catch (e: Exception) {
        Log.e("convertToGrayscale", "Error converting to grayscale", e)
        null
    }
}

private fun saveCopyToDownloads(context: android.content.Context, uri: Uri) {
    try {
        Log.w("saveCopyToDownloads", "Starting to save image to downloads")
        val resolver = context.contentResolver
        val name = "processed_${System.currentTimeMillis()}.jpg"
        Log.w("saveCopyToDownloads", "Generated filename: $name")
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Download/")
        }
        val outUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        Log.w("saveCopyToDownloads", "Created output URI: $outUri")
        val input = resolver.openInputStream(uri)
        val output = outUri?.let { resolver.openOutputStream(it) }
        if (input != null && output != null) {
            Log.w("saveCopyToDownloads", "Copying file data")
            input.copyTo(output)
            input.close(); output.close()
            Log.w("saveCopyToDownloads", "File saved successfully")
        } else {
            Log.e("saveCopyToDownloads", "Failed to open input or output stream")
        }
    } catch (e: Exception) {
        Log.e("saveCopyToDownloads", "Error saving file", e)
        e.printStackTrace()
    }
}

/**
 * 递归拷贝Assets目录到应用内部存储
 * 
 * 源目录：Android应用的assets目录
 * - 位置：app/src/main/assets/
 * - 特点：只读，打包在APK中，无法修改
 * - 访问方式：通过AssetManager访问
 * 
 * 目标目录：应用内部存储的files目录
 * - 位置：/data/data/包名/files/
 * - 特点：可读写，应用私有，卸载时会删除
 * - 访问方式：通过Context.filesDir获取
 * 
 * @param context Android上下文，用于获取AssetManager和filesDir
 * @param assetDir assets中的源目录路径（相对路径，如"resources"）
 * @param outDir 目标目录的File对象（绝对路径，如/data/data/包名/files/resources）
 */
fun copyAssetDirToFiles(context: android.content.Context, assetDir: String, outDir: java.io.File) {
    // 获取AssetManager，用于访问assets目录中的资源
    val am = context.assets
    
    // 列出assets中指定目录下的所有文件和子目录
    // 如果目录不存在或为空，直接返回
    val list = am.list(assetDir) ?: return
    
    // 确保目标目录存在，如果不存在则创建
    if (!outDir.exists()) outDir.mkdirs()
    
    // 遍历assets目录中的每个文件/子目录
    for (name in list) {
        // 构建assets中的完整路径
        // 例如：assetDir="resources", name="models" -> assetPath="resources/models"
        val assetPath = if (assetDir.isEmpty()) name else "$assetDir/$name"
        
        // 构建目标文件/目录的完整路径
        // 例如：outDir="/data/data/包名/files/resources", name="models" 
        // -> out="/data/data/包名/files/resources/models"
        val out = java.io.File(outDir, name)
        
        // 检查当前项是否为目录（通过尝试列出其子项来判断）
        val children = am.list(assetPath)
        
        if (children != null && children.isNotEmpty()) {
            // 如果是目录且非空，递归拷贝子目录
            // 例如：拷贝assets/resources/models/到/data/data/包名/files/resources/models/
            copyAssetDirToFiles(context, assetPath, out)
        } else {
            // 如果是文件，直接拷贝文件内容
            // 从assets中打开输入流，创建目标文件的输出流，然后拷贝数据
            am.open(assetPath).use { input ->
                java.io.FileOutputStream(out).use { output -> input.copyTo(output) }
            }
        }
    }
}

/**
 * 确保资源文件已从assets拷贝到内部存储
 * 
 * 拷贝路径详解：
 * 源：assets/resources/ （APK中的只读资源目录）
 * 目标：/data/data/com.example.imageapp/files/resources/ （应用可写的内部存储）
 * 
 * 为什么需要拷贝：
 * 1. assets中的文件是只读的，nndeploy等底层库可能需要可写权限
 * 2. 某些库需要真实的文件路径，而不是assets的虚拟路径
 * 3. 提高访问性能，避免每次都从APK中解压读取
 * 
 * @param context Android上下文
 */
fun ensureResourcesReady(context: android.content.Context) {
    // 创建标记文件，用于检查资源是否已经拷贝过
    // 路径：/data/data/com.example.imageapp/files/resources/.installed
    val marker = java.io.File(context.filesDir, "resources/.installed")
    
    if (!marker.exists()) {
        // 如果标记文件不存在，说明还未拷贝过资源
        // 执行拷贝：从assets/resources/拷贝到/data/data/包名/files/resources/
        copyAssetDirToFiles(context, "resources", java.io.File(context.filesDir, "resources"))
        
        // 确保标记文件的父目录存在
        marker.parentFile?.mkdirs()
        
        // 创建标记文件，写入当前时间戳作为拷贝完成的标记
        marker.writeText(System.currentTimeMillis().toString())
    }
    // 如果标记文件存在，说明资源已经拷贝过，直接跳过
}

fun getExternalRoot(context: android.content.Context): java.io.File {
    // getExternalFilesDir(null) 会自动创建目录如果不存在
    // 返回路径：/sdcard/Android/data/<pkg>/files
    // 如果外部存储不可用（如SD卡被移除），会返回null
    return requireNotNull(context.getExternalFilesDir(null)) // /sdcard/Android/data/<pkg>/files
}

fun ensureExternalResourcesReady(context: android.content.Context): java.io.File {
    val root = getExternalRoot(context)
    val resDir = java.io.File(root, "resources")
    val marker = java.io.File(resDir, ".installed")
    if (!marker.exists()) {
        copyAssetDirToFiles(context, "resources", resDir) // 复用你已有的拷贝函数
        marker.parentFile?.mkdirs()
        marker.writeText(System.currentTimeMillis().toString())
    }
    return resDir
}
