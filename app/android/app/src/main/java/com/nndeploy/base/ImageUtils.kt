// /home/always/github/public/nndeploy/app/android/app/src/main/java/com/nndeploy/base/ImageUtils.kt
package com.nndeploy.base

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import java.io.File
import java.io.FileOutputStream

/**
 * 图像处理工具类
 */
object ImageUtils {
    
    /**
     * 创建图片选择启动器
     */
    fun createImagePickerLauncher(
        activity: ComponentActivity,
        onImageSelected: (Uri?) -> Unit
    ): ActivityResultLauncher<String> {
        return activity.registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            onImageSelected(uri)
        }
    }
    
    /**
     * 预处理图片 - 格式转换、尺寸调整等
     */
    fun preprocessImage(context: Context, inputUri: Uri): Uri {
        return try {
            Log.w("ImageUtils", "Starting image preprocessing")
            
            // 读取原始图片
            val originalBitmap = loadBitmapFromUri(context, inputUri)
            
            // 图片预处理：调整尺寸、格式转换等
            val processedBitmap = processImageBitmap(originalBitmap)
            
            // 保存处理后的图片
            val outputFile = File(context.cacheDir, "processed_image_${System.currentTimeMillis()}.jpg")
            saveBitmapToFile(processedBitmap, outputFile)
            
            // 释放资源
            if (originalBitmap != processedBitmap) {
                originalBitmap.recycle()
            }
            processedBitmap.recycle()
            
            Uri.fromFile(outputFile)
        } catch (e: Exception) {
            Log.e("ImageUtils", "Image preprocessing failed", e)
            inputUri // 如果预处理失败，返回原始URI
        }
    }
    
    /**
     * 从URI加载Bitmap
     */
    private fun loadBitmapFromUri(context: Context, uri: Uri): Bitmap {
        val resolver = context.contentResolver
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(resolver, uri)
            ImageDecoder.decodeBitmap(source)
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(resolver, uri)
        }
    }
    
    /**
     * 处理图片Bitmap - 调整尺寸、旋转等
     */
    private fun processImageBitmap(bitmap: Bitmap): Bitmap {
        val maxSize = 1024 // 最大尺寸限制
        
        return if (bitmap.width > maxSize || bitmap.height > maxSize) {
            // 计算缩放比例
            val scale = minOf(
                maxSize.toFloat() / bitmap.width,
                maxSize.toFloat() / bitmap.height
            )
            
            val newWidth = (bitmap.width * scale).toInt()
            val newHeight = (bitmap.height * scale).toInt()
            
            // 创建缩放后的bitmap
            Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        } else {
            bitmap
        }
    }
    
    /**
     * 保存Bitmap到文件
     */
    private fun saveBitmapToFile(bitmap: Bitmap, file: File) {
        FileOutputStream(file).use { output ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, output)
        }
    }
    
    /**
     * 旋转图片
     */
    fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
        
    /**
     * 获取图片信息
     */
    fun getImageInfo(context: Context, uri: Uri): ImageInfo? {
        return try {
            val bitmap = loadBitmapFromUri(context, uri)
            val info = ImageInfo(
                width = bitmap.width,
                height = bitmap.height,
                size = bitmap.byteCount.toLong()
            )
            bitmap.recycle()
            info
        } catch (e: Exception) {
            Log.e("ImageUtils", "Failed to get image info", e)
            null
        }
    }
}

/**
 * 图片信息数据类
 */
data class ImageInfo(
    val width: Int,
    val height: Int,
    val size: Long
)