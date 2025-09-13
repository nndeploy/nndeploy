// /home/always/github/public/nndeploy/app/android/app/src/main/java/com/nndeploy/base/CameraUtils.kt
package com.nndeploy.base

import android.content.Context
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.ComponentActivity
import android.net.Uri
import android.provider.MediaStore
import android.content.ContentValues
import java.io.File

/**
 * 摄像头工具类
 */
object CameraUtils {
    
    /**
     * 创建相机拍照启动器
     */
    fun createCameraLauncher(
        activity: ComponentActivity,
        onPhotoTaken: (Uri?, Boolean) -> Unit
    ): ActivityResultLauncher<Uri> {
        return activity.registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
            onPhotoTaken(null, success)
        }
    }
    
    /**
     * 创建视频录制启动器
     */
    fun createVideoCaptureLauncher(
        activity: ComponentActivity,
        onVideoRecorded: (Uri?, Boolean) -> Unit
    ): ActivityResultLauncher<Uri> {
        return activity.registerForActivityResult(ActivityResultContracts.CaptureVideo()) { success ->
            onVideoRecorded(null, success)
        }
    }
    
    /**
     * 创建临时图片文件用于拍照
     */
    fun createTempImageFile(context: Context): File {
        val fileName = "photo_${System.currentTimeMillis()}.jpg"
        return File(context.cacheDir, fileName)
    }
    
    /**
     * 创建临时视频文件用于录像
     */
    fun createTempVideoFile(context: Context): File {
        val fileName = "video_${System.currentTimeMillis()}.mp4"
        return File(context.cacheDir, fileName)
    }
    
    /**
     * 创建拍照URI（使用FileProvider）
     */
    fun createPhotoUri(context: Context): Uri {
        val photoFile = createTempImageFile(context)
        return androidx.core.content.FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            photoFile
        )
    }
    
    /**
     * 创建录像URI（使用FileProvider）
     */
    fun createVideoUri(context: Context): Uri {
        val videoFile = createTempVideoFile(context)
        return androidx.core.content.FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            videoFile
        )
    }
    
    /**
     * 保存图片到媒体库
     */
    fun saveImageToMediaStore(context: Context, sourceUri: Uri): Uri? {
        return try {
            val resolver = context.contentResolver
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, "photo_${System.currentTimeMillis()}.jpg")
                put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/NNDeploy/")
            }
            
            val mediaUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            
            mediaUri?.let { uri ->
                resolver.openInputStream(sourceUri)?.use { input ->
                    resolver.openOutputStream(uri)?.use { output ->
                        input.copyTo(output)
                    }
                }
                uri
            }
        } catch (e: Exception) {
            null
        }
    }
    
    /**
     * 保存视频到媒体库
     */
    fun saveVideoToMediaStore(context: Context, sourceUri: Uri): Uri? {
        return try {
            val resolver = context.contentResolver
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, "video_${System.currentTimeMillis()}.mp4")
                put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/NNDeploy/")
            }
            
            val mediaUri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, contentValues)
            
            mediaUri?.let { uri ->
                resolver.openInputStream(sourceUri)?.use { input ->
                    resolver.openOutputStream(uri)?.use { output ->
                        input.copyTo(output)
                    }
                }
                uri
            }
        } catch (e: Exception) {
            null
        }
    }
}