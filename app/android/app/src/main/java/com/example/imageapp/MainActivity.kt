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

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
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
            selected = false, onClick = { nav.navigate("home") },
            icon = { Icon(Icons.Default.Description, contentDescription = "首页") }, label = { Text("首页") }
        )
        NavigationBarItem(
            selected = false, onClick = { nav.navigate("process") },
            icon = { Icon(Icons.Default.History, contentDescription = "处理") }, label = { Text("处理") }
        )
        NavigationBarItem(
            selected = false, onClick = { nav.navigate("history") },
            icon = { Icon(Icons.Default.History, contentDescription = "历史") }, label = { Text("历史") }
        )
    }
}

@Composable
fun UploadScreen(nav: NavHostController, vm: AppVM) {
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        vm.pickedImage = uri
        if (uri != null) nav.navigate("process")
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
                Button(onClick = { launcher.launch("image/*") }) {
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
            FilterChip(label = "分割", selected = selected == "分割") { selected = "分割" }
            Spacer(Modifier.width(12.dp))
            FilterChip(label = "抠脸", selected = selected == "抠脸") { selected = "抠脸" }
        }
        Spacer(Modifier.height(12.dp))
        Button(
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(28.dp),
            onClick = {
                scope.launch {
                    try {
                        val runner = GraphRunner()
                        // runner.setJsonFile(false)
                        runner.setTimeProfile(true)
                        // runner.setDebug(false)

                        // // TODO: 替换为真实分割图 JSON
                        // val graphJson = "{" +
                        //         "\"name_\":\"seg_demo\"," +
                        //         "\"node_repository_\":[]" +
                        //         "}"
                        val workflowPath = "resources/workflow/ClassificationResNetMnn.json"
                        val graphJson = context.assets.open(workflowPath).bufferedReader().use { it.readText() }
                        val ok = runner.run(workflowPath, "seg_demo", "task_${'$'}{System.currentTimeMillis()}")
//                        val filePath = File(context.filesDir, "resources/workflow/ClassificationResNetMnn.json").absolutePath
                        // val ok = runner.run(filePath, "seg_demo", "task_${System.currentTimeMillis()}")
                        runner.close()

                        if (ok) {
                            Toast.makeText(context, "分割执行成功", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(context, "分割执行失败", Toast.LENGTH_SHORT).show()
                        }
                    } catch (e: Throwable) {
                        Toast.makeText(context, "JNI 异常: ${'$'}{e.message}", Toast.LENGTH_SHORT).show()
                    } finally {
                        vm.processedImage = uri
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
                    resultUri?.let { saveCopyToDownloads(context, it) }
                },
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF183D8C)),
                modifier = Modifier.weight(1f).height(52.dp)
            ) { Text("下载图片") }
            Button(
                onClick = {
                    // Share
                    resultUri?.let {
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
            onClick = { nav.navigate("process") },
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
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("历史（占位）")
    }
}

private fun saveCopyToDownloads(context: android.content.Context, uri: Uri) {
    try {
        val resolver = context.contentResolver
        val name = "processed_${System.currentTimeMillis()}.jpg"
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Download/")
        }
        val outUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        val input = resolver.openInputStream(uri)
        val output = outUri?.let { resolver.openOutputStream(it) }
        if (input != null && output != null) {
            input.copyTo(output)
            input.close(); output.close()
        }
    } catch (e: Exception) {
        e.printStackTrace()
    }
}
