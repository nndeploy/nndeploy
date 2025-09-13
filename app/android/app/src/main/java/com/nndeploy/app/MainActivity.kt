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
    Log.w("App", "App composable initialized")
    Scaffold(
        bottomBar = { BottomBar(nav) }
    ) { inner ->
        NavHost(
            navController = nav,
            startDestination = "ai",
            modifier = Modifier.padding(inner)
        ) {
            // AI工具页面
            composable("ai") { AIScreen(nav) }
            composable("ai_process/{algorithmId}") { backStackEntry ->
                val algorithmId = backStackEntry.arguments?.getString("algorithmId") ?: ""
                AIProcessScreen(nav, algorithmId)
            }
            composable("ai_result") { AIResultScreen(nav) }
            
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



