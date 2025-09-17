// /home/always/github/public/nndeploy/app/android/app/src/main/java/com/nndeploy/base/VideoUtils.kt
package com.nndeploy.base

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * 视频处理工具类
 */
object VideoUtils {
    
    /**
     * 创建视频选择启动器
     */
    fun createVideoPickerLauncher(
        activity: ComponentActivity,
        onVideoSelected: (Uri?) -> Unit
    ): ActivityResultLauncher<String> {
        return activity.registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            onVideoSelected(uri)
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
     * 预处理视频 - 格式转换、压缩等
     */
    fun preprocessVideo(context: Context, inputUri: Uri): Uri {
        return try {
            Log.w("VideoUtils", "Starting video preprocessing")
            
            // 获取视频信息
            val videoInfo = getVideoInfo(context, inputUri)
            Log.w("VideoUtils", "Video info: $videoInfo")
            
            // 如果视频太大，可以进行压缩处理
            if (videoInfo != null && shouldCompressVideo(videoInfo)) {
                compressVideo(context, inputUri)
            } else {
                // 直接复制到缓存目录
                copyVideoToCache(context, inputUri)
            }
        } catch (e: Exception) {
            Log.e("VideoUtils", "Video preprocessing failed", e)
            inputUri // 如果预处理失败，返回原始URI
        }
    }
    
    /**
     * 获取视频信息
     */
    fun getVideoInfo(context: Context, uri: Uri): VideoInfo? {
        return try {
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(context, uri)
            
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val bitrate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE)?.toIntOrNull() ?: 0
            
            // 获取文件大小
            val size = try {
                context.contentResolver.openInputStream(uri)?.use { it.available().toLong() } ?: 0L
            } catch (e: Exception) {
                0L
            }
            
            retriever.release()
            
            VideoInfo(
                width = width,
                height = height,
                duration = duration,
                bitrate = bitrate,
                size = size
            )
        } catch (e: Exception) {
            Log.e("VideoUtils", "Failed to get video info", e)
            null
        }
    }
    
    /**
     * 判断是否需要压缩视频
     */
    private fun shouldCompressVideo(videoInfo: VideoInfo): Boolean {
        val maxSize = 50 * 1024 * 1024 // 50MB
        val maxResolution = 1920 // 1080p
        
        return videoInfo.size > maxSize || 
               videoInfo.width > maxResolution || 
               videoInfo.height > maxResolution
    }
    
    /**
     * 压缩视频（简单实现，实际项目中可能需要使用FFmpeg等库）
     */
    private fun compressVideo(context: Context, inputUri: Uri): Uri {
        // 这里是一个简化的实现，实际项目中可能需要使用FFmpeg等库进行视频压缩
        // 目前只是复制文件
        return copyVideoToCache(context, inputUri)
    }
    
    /**
     * 复制视频到缓存目录
     */
    private fun copyVideoToCache(context: Context, inputUri: Uri): Uri {
        val outputFile = File(context.cacheDir, "processed_video_${System.currentTimeMillis()}.mp4")
        
        context.contentResolver.openInputStream(inputUri)?.use { input ->
            FileOutputStream(outputFile).use { output ->
                input.copyTo(output)
            }
        }
        
        return Uri.fromFile(outputFile)
    }
    
    /**
     * 提取视频缩略图
     */
    fun extractVideoThumbnail(context: Context, uri: Uri): android.graphics.Bitmap? {
        return try {
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(context, uri)
            val bitmap = retriever.getFrameAtTime(0)
            retriever.release()
            bitmap
        } catch (e: Exception) {
            Log.e("VideoUtils", "Failed to extract video thumbnail", e)
            null
        }
    }
    
    /**
     * 获取视频时长（毫秒）
     */
    fun getVideoDuration(context: Context, uri: Uri): Long {
        return try {
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(context, uri)
            val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            retriever.release()
            duration
        } catch (e: Exception) {
            Log.e("VideoUtils", "Failed to get video duration", e)
            0L
        }
    }
    
    /**
     * 检查视频格式是否支持
     */
    fun isVideoFormatSupported(context: Context, uri: Uri): Boolean {
        return try {
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(context, uri)
            val mimeType = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_MIMETYPE)
            retriever.release()
            
            // 支持的视频格式
            val supportedFormats = listOf("video/mp4", "video/avi", "video/mov", "video/3gpp")
            mimeType in supportedFormats
        } catch (e: Exception) {
            Log.e("VideoUtils", "Failed to check video format", e)
            false
        }
    }
}

/**
 * 视频信息数据类
 */
data class VideoInfo(
    val width: Int,
    val height: Int,
    val duration: Long, // 毫秒
    val bitrate: Int,
    val size: Long // 字节
)