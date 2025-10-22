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
    val sharedAIViewModel: AIViewModel = viewModel() // Create shared AI ViewModel
    
    Log.w("App", "App composable initialized")
    Scaffold(
        bottomBar = { BottomBar(nav) }
    ) { inner ->
        NavHost(
            navController = nav,
            startDestination = "ai",
            modifier = Modifier.padding(inner)
        ) {
            // AI algorithm page - pass shared ViewModel
            composable("ai") { 
                AIScreen(nav, sharedAIViewModel) 
            }
            // // AI algorithm processing page route - receive algorithm ID parameter
            // // Path format: "ai_process/{algorithmId}" where algorithmId is dynamic parameter
            // composable("ai_process/{algorithmId}") { backStackEntry ->
            //     // Extract algorithm ID from navigation parameters, use empty string as default if failed
            //     val algorithmId = backStackEntry.arguments?.getString("algorithmId") ?: ""
            //     // Call CV processing page, pass navigation controller, algorithm ID and shared AI ViewModel
            //     CVProcessScreen(nav, algorithmId, sharedAIViewModel)
            // }
            // Modify route in NavHost in MainActivity.kt
            composable("ai_process/{algorithmId}") { backStackEntry ->
                val algorithmId = backStackEntry.arguments?.getString("algorithmId") ?: ""
                val algorithm = AlgorithmFactory.getAlgorithmsById(sharedAIViewModel.availableAlgorithms, algorithmId)
                
                // Choose different processing pages based on algorithm's processFunction
                when (algorithm?.processFunction) {
                    "processPromptInPromptOut" -> {
                        // Intelligent chat algorithms use LlmChatProcessScreen
                        LlmChatProcessScreen(nav, algorithmId, sharedAIViewModel)
                    }
                    "processImageInImageOut" -> {
                        // Image processing algorithms use CVProcessScreen
                        CVProcessScreen(nav, algorithmId, sharedAIViewModel)
                    }
                    else -> {
                        // Default to CVProcessScreen
                        CVProcessScreen(nav, algorithmId, sharedAIViewModel)
                    }
                }
            }
            
            // AI algorithm result display page route
            // Used to display results after algorithm processing is complete
            composable("ai_result") { 
                // Call CV result page, pass navigation controller and shared AI ViewModel
                CVResultScreen(nav, sharedAIViewModel) 
            }
            
            // Mine page
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
            icon = { Icon(Icons.Default.SmartToy, contentDescription = "AI Tools") }, 
            label = { Text("AI Tools") }
        )
        NavigationBarItem(
            selected = false, onClick = { 
                Log.w("BottomBar", "Navigate to mine")
                nav.navigate("mine") 
            },
            icon = { Icon(Icons.Default.Person, contentDescription = "Mine") }, 
            label = { Text("Mine") }
        )
    }
}


