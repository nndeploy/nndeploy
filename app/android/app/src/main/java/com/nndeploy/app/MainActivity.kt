package com.nndeploy.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.SmartToy
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.nndeploy.app.ui.theme.AppTheme
import android.util.Log
import android.net.Uri
import com.nndeploy.ai.AlgorithmFactory

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
    // ViewModel for app state management
}

@Composable
fun App() {
    val nav = rememberNavController()
    val vm: AppVM = viewModel()
    val sharedAIViewModel: AIViewModel = viewModel() // 创建共享的AI ViewModel
    
    Log.w("App", "App composable initialized")
    Scaffold(
        bottomBar = { BottomBar(nav) }
    ) { inner ->
        NavHost(
            navController = nav,
            startDestination = "ai",
            modifier = Modifier.padding(inner)
        ) {
            // AI算法页面 - 传递共享ViewModel
            composable("ai") { 
                AIScreen(nav, sharedAIViewModel) 
            }
            // // AI算法处理页面路由 - 接收算法ID参数
            // // 路径格式: "ai_process/{algorithmId}" 其中algorithmId是动态参数
            // composable("ai_process/{algorithmId}") { backStackEntry ->
            //     // 从导航参数中提取算法ID，如果获取失败则使用空字符串作为默认值
            //     val algorithmId = backStackEntry.arguments?.getString("algorithmId") ?: ""
            //     // 调用CV处理页面，传入导航控制器、算法ID和共享的AI ViewModel
            //     CVProcessScreen(nav, algorithmId, sharedAIViewModel)
            // }
            // 在 MainActivity.kt 的 NavHost 中修改路由
            composable("ai_process/{algorithmId}") { backStackEntry ->
                val algorithmId = backStackEntry.arguments?.getString("algorithmId") ?: ""
                val algorithm = AlgorithmFactory.getAlgorithmsById(sharedAIViewModel.availableAlgorithms, algorithmId)
                
                // 根据算法的processFunction选择不同的处理页面
                when (algorithm?.processFunction) {
                    "processPromptInPromptOut" -> {
                        // 智能对话类算法使用LlmChatProcessScreen
                        LlmChatProcessScreen(nav, algorithmId, sharedAIViewModel)
                    }
                    "processImageInImageOut" -> {
                        // 图像处理类算法使用CVProcessScreen
                        CVProcessScreen(nav, algorithmId, sharedAIViewModel)
                    }
                    else -> {
                        // 默认使用CVProcessScreen
                        CVProcessScreen(nav, algorithmId, sharedAIViewModel)
                    }
                }
            }
            
            // AI算法结果展示页面路由
            // 用于显示算法处理完成后的结果
            composable("ai_result") { 
                // 调用CV结果页面，传入导航控制器和共享的AI ViewModel
                CVResultScreen(nav, sharedAIViewModel) 
            }
            
            // 我的页面
            composable("mine") { MineScreen(nav) }
        }
    }
}

@Composable
fun BottomBar(nav: NavHostController) {
    NavigationBar {
        NavigationBarItem(
            selected = false, onClick = { 
                Log.w("BottomBar", "Navigate to AI")
                nav.navigate("ai") 
            },
            icon = { Icon(Icons.Default.SmartToy, contentDescription = "AI工具") }, 
            label = { Text("AI工具") }
        )
        NavigationBarItem(
            selected = false, onClick = { 
                Log.w("BottomBar", "Navigate to mine")
                nav.navigate("mine") 
            },
            icon = { Icon(Icons.Default.Person, contentDescription = "我的") }, 
            label = { Text("我的") }
        )
    }
}



