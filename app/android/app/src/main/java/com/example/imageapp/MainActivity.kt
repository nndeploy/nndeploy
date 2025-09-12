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
                            Log.w("ProcessScreen", "Initializing GraphRunner")
                            // 在/data/local/tmp下创建一个test_app.txt文件，并写入当前时间
                            // context.filesDir 指向应用的私有内部存储目录
                            // 通常位于: /data/data/com.example.imageapp/files/
                            // 这个目录只有当前应用可以访问，系统会在应用卸载时自动清理
                            val testFile = File(context.filesDir, "test_app.txt")
                            testFile.writeText("Initializing GraphRunner: ${System.currentTimeMillis()}\nFilesDir path: ${context.filesDir.absolutePath}")
                            Log.w("ProcessScreen", "Test file created at: ${testFile.absolutePath}")

                            val runner = GraphRunner()
                            runner.setJsonFile(false)
                            runner.setTimeProfile(true)
                            // runner.setDebug(false)

                            // val workflowPath = "resources/workflow/ClassificationResNetMnn.json"
                            // val workflowPath = "class_mnn_test.json"
                            val workflowPath = "demo.json"
                            val absolutePath = File(context.filesDir, workflowPath).absolutePath
                            Log.w("ProcessScreen", "Loading workflow from: $workflowPath")
                            Log.w("ProcessScreen", "Absolute path: $absolutePath")
                            val graphJson: String = context.assets.open(workflowPath).bufferedReader().use { it.readText() }
                            Log.w("ProcessScreen", "Workflow JSON loaded, length: ${graphJson.length}")
                            Log.w("ProcessScreen", "GraphJSON content: $graphJson")
                            
                            val taskId = "task_${System.currentTimeMillis()}"
                            Log.w("ProcessScreen", "Running task: $taskId")
                            // val ok = runner.run(absolutePath, "seg_demo", taskId)
                            val ok = runner.run(graphJson, "seg_demo", taskId)
                            Log.w("ProcessScreen", "GraphRunner closed")
                            vm.processedImage = uri
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

fun copyAssetDirToFiles(context: android.content.Context, assetDir: String, outDir: java.io.File) {
    val am = context.assets
    val list = am.list(assetDir) ?: return
    if (!outDir.exists()) outDir.mkdirs()
    for (name in list) {
        val assetPath = if (assetDir.isEmpty()) name else "$assetDir/$name"
        val out = java.io.File(outDir, name)
        val children = am.list(assetPath)
        if (children != null && children.isNotEmpty()) {
            copyAssetDirToFiles(context, assetPath, out)
        } else {
            am.open(assetPath).use { input ->
                java.io.FileOutputStream(out).use { output -> input.copyTo(output) }
            }
        }
    }
}

fun ensureResourcesReady(context: android.content.Context) {
    val marker = java.io.File(context.filesDir, "resources/.installed")
    if (!marker.exists()) {
        copyAssetDirToFiles(context, "resources", java.io.File(context.filesDir, "resources"))
        marker.parentFile?.mkdirs()
        marker.writeText(System.currentTimeMillis().toString())
    }
}
