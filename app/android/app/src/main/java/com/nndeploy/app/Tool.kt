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
 * AI Page ViewModel
 */
class AIViewModel : ViewModel() {
    var selectedAlgorithm by mutableStateOf<AIAlgorithm?>(null)
    var inputUri by mutableStateOf<Uri?>(null)
    var outputUri by mutableStateOf<Uri?>(null)
    var isProcessing by mutableStateOf(false)
    
    // Available AI algorithms list
    val availableAlgorithms = AlgorithmFactory.createDefaultAlgorithms()
}

/**
 * AI Algorithm Home Page
 */
@Composable
fun AIScreen(nav: NavHostController, sharedViewModel: AIViewModel = viewModel()) {
    val vm: AIViewModel = sharedViewModel
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF8FAFC))
    ) {
        // Title bar
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White)
                .padding(16.dp)
        ) {
            Text(
                text = "nndeploy Algorithm Center",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF1E3A8A)
            )
        }
        
        // Algorithm list
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Group by category
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
 * AI Algorithm Card
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
            // Algorithm icon
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
            
            // Algorithm information
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
                
                // Supported input type tags
                Row {
                    algorithm.inputType.forEach { type ->
                        InputTypeChip(type)
                        Spacer(modifier = Modifier.width(4.dp))
                    }
                }
            }
            
            // Arrow icon
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
 * Input Type Chip
 */
@Composable
fun InputTypeChip(inputType: InOutType) {
    val (text, color) = when (inputType) {
        InOutType.IMAGE -> "Image" to Color(0xFF10B981)
        InOutType.VIDEO -> "Video" to Color(0xFFF59E0B)
        InOutType.CAMERA -> "Camera" to Color(0xFFEF4444)
        InOutType.AUDIO -> "Audio" to Color(0xFFFF6B6B)
        InOutType.TEXT -> "Text" to Color(0xFF4ECDC4)
        InOutType.PROMPT -> "Prompt" to Color(0xFF8B5CF6)
        InOutType.ALL -> "All" to Color(0xFF06B6D4)
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
 * AI Algorithm Processing Page
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
    
    // Find the corresponding algorithm
    val algorithm = AlgorithmFactory.getAlgorithmsById(vm.availableAlgorithms, algorithmId)
    
    // Image picker
    val imagePickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        vm.inputUri = uri
    }
    
    // Video picker
    val videoPickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        vm.inputUri = uri
    }
    
    // Camera
    val cameraLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            // Handle photo taken successfully
            Log.w("CVProcessScreen", "Photo taken successfully")
        }
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF8FAFC))
    ) {
        // Top bar
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
                    contentDescription = "Back"
                )
            }
            Text(
                text = algorithm?.name ?: "AI Processing",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF1E3A8A),
                modifier = Modifier.weight(1f)
            )
        }
        
        // Input area
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
                        contentDescription = "Input content",
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
                            text = "Select input content",
                            fontSize = 16.sp,
                            color = Color(0xFF6B7280)
                        )
                    }
                }
            }
        }
        
        // Input selection buttons
        algorithm?.let { algo ->
            InputSelectionButtons(
                inputTypes = algo.inputType,
                onImageSelect = { imagePickerLauncher.launch("image/*") },
                onVideoSelect = { videoPickerLauncher.launch("video/*") },
                onCameraPhoto = { 
                    // Create temporary file for photo
                    val photoUri = CameraUtils.createPhotoUri(context)
                    vm.inputUri = photoUri
                    cameraLauncher.launch(photoUri)
                },
                onCameraVideo = {
                    // Video recording function implementation
                    Toast.makeText(context, "Video recording feature to be implemented", Toast.LENGTH_SHORT).show()
                }
            )
        }
        
        // Process button
        Button(
            onClick = {
                vm.inputUri?.let { uri ->
                    scope.launch {
                        vm.isProcessing = true
                        // Check if algorithm exists
                        if (algorithm == null) {
                            Toast.makeText(context, "Algorithm $algorithmId does not exist", Toast.LENGTH_LONG).show()
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
                } ?: Toast.makeText(context, "Please select input content first", Toast.LENGTH_SHORT).show()
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
                Text("Processing...")
            } else {
                Text("Start Processing", fontSize = 16.sp)
            }
        }
    }
}

/**
 * Input Selection Button Group
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
            text = "Select input method",
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF374151),
            modifier = Modifier.padding(bottom = 12.dp)
        )
        
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Show corresponding buttons based on algorithm supported input types
            if (inputTypes.contains(InOutType.IMAGE)) {
                OutlinedButton(
                    onClick = onImageSelect,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.Photo, null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("Gallery")
                }
                OutlinedButton(
                    onClick = onCameraPhoto,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.CameraAlt, null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("Camera")
                }
            }
        }
    }
}

/**
 * AI Processing Result Page
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
        // Top bar
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
                    contentDescription = "Back"
                )
            }
            Text(
                text = "Processing Result",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF1E3A8A),
                modifier = Modifier.weight(1f)
            )
        }
        
        // Result display area
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
                        contentDescription = "Processing result",
                        modifier = Modifier.fillMaxWidth(0.9f)
                    )
                } else {
                    Text(
                        text = "No processing result",
                        fontSize = 16.sp,
                        color = Color(0xFF6B7280)
                    )
                }
            }
        }
        
        // Action buttons
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
                            Toast.makeText(context, "Saved successfully", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(context, "Save failed", Toast.LENGTH_SHORT).show()
                        }
                    }
                },
                modifier = Modifier.weight(1f)
            ) {
                Icon(Icons.Default.Download, null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(4.dp))
                Text("Save")
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
                        context.startActivity(android.content.Intent.createChooser(shareIntent, "Share result"))
                    }
                },
                modifier = Modifier.weight(1f)
            ) {
                Icon(Icons.Default.Share, null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(4.dp))
                Text("Share")
            }
        }
        
        // Continue processing button
        Button(
            onClick = { nav.navigate("ai") },
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF10B981)),
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp)
                .height(52.dp),
            shape = RoundedCornerShape(26.dp)
        ) {
            Text("Continue processing other algorithms", fontSize = 16.sp)
        }
    }
}

/**
 * Determine input media type based on URI
 */
private fun determineInputType(uri: Uri, context: Context): com.nndeploy.ai.InputMediaType {
    return try {
        val mimeType = context.contentResolver.getType(uri)
        when {
            mimeType?.startsWith("image/") == true -> com.nndeploy.ai.InputMediaType.IMAGE
            mimeType?.startsWith("video/") == true -> com.nndeploy.ai.InputMediaType.VIDEO
            else -> com.nndeploy.ai.InputMediaType.IMAGE // Default to image
        }
    } catch (e: Exception) {
        com.nndeploy.ai.InputMediaType.IMAGE // Default to image on exception
    }
}

/**
 * LLM Chat Processing Page
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
    
    // Find the corresponding algorithm
    val algorithm = AlgorithmFactory.getAlgorithmsById(vm.availableAlgorithms, algorithmId)
    
    // Chat message state
    var messages by remember { mutableStateOf(listOf<ChatMessage>()) }
    var inputText by remember { mutableStateOf("") }
    var isTyping by remember { mutableStateOf(false) }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF8FAFC))
    ) {
        // Top bar
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
                    contentDescription = "Back"
                )
            }
            Text(
                text = algorithm?.name ?: "AI Chat",
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
                    contentDescription = "Clear chat"
                )
            }
        }
        
        // Chat message area
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
                                text = "Start conversation with AI",
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color(0xFF374151)
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Enter your question, AI will help you",
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
            
            // Typing indicator
            if (isTyping) {
                item {
                    TypingIndicator()
                }
            }
        }
        
        // Input area
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
                    placeholder = { Text("Enter your question...") },
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
                            // 1. Save current input text to local variable
                            val currentInput = inputText
                            
                            // 2. Immediately show user message in chat interface
                            val userMessage = ChatMessage(
                                content = currentInput,
                                isUser = true,
                                timestamp = System.currentTimeMillis()
                            )
                            messages = messages + userMessage
                            
                            // 3. Immediately clear input field to improve user experience
                            inputText = ""
                            
                            // 4. Launch coroutine to handle AI response
                            scope.launch {
                                isTyping = true
                                try {
                                    // Check if algorithm exists
                                    if (algorithm == null) {
                                        Toast.makeText(context, "Algorithm $algorithmId does not exist", Toast.LENGTH_LONG).show()
                                        return@launch
                                    }
                                    
                                    Log.d("LlmChatProcessScreen", "currentInput: $currentInput")
                                    // 5. Use saved input text for processing
                                    val result = PromptInPromptOut.processPromptInPromptOut(context, currentInput, algorithm)
                                    
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
                                                content = "Sorry, an error occurred: ${result.message}",
                                                isUser = false,
                                                timestamp = System.currentTimeMillis(),
                                                isError = true
                                            )
                                            messages = messages + errorMessage
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e("LlmChatProcessScreen", "AI processing failed", e)
                                    val errorMessage = ChatMessage(
                                        content = "Sorry, an unknown error occurred: ${e.message}",
                                        isUser = false,
                                        timestamp = System.currentTimeMillis(),
                                        isError = true
                                    )
                                    messages = messages + errorMessage
                                } finally {
                                    isTyping = false
                                }
                            }
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
                        contentDescription = "Send",
                        tint = if (inputText.isNotBlank() && !isTyping) Color.White else Color(0xFF9CA3AF)
                    )
                }
            }
        }
    }
}

/**
 * Chat Message Data Class
 */
data class ChatMessage(
    val content: String,
    val isUser: Boolean,
    val timestamp: Long,
    val isError: Boolean = false
)

/**
 * Chat Message Item Component
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
 * Typing Indicator
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
