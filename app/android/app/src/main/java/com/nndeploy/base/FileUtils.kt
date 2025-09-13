// /home/always/github/public/nndeploy/app/android/app/src/main/java/com/nndeploy/base/FileUtils.kt
package com.nndeploy.base

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * 文件处理工具类
 */
object FileUtils {
    
    /**
     * 保存图片到下载目录
     */
    fun saveCopyToDownloads(context: Context, uri: Uri): Boolean {
        return try {
            Log.w("FileUtils", "Starting to save image to downloads")
            val resolver = context.contentResolver
            val name = "processed_${System.currentTimeMillis()}.jpg"
            Log.w("FileUtils", "Generated filename: $name")
            
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Download/")
            }
            
            val outUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            Log.w("FileUtils", "Created output URI: $outUri")
            
            val input = resolver.openInputStream(uri)
            val output = outUri?.let { resolver.openOutputStream(it) }
            
            if (input != null && output != null) {
                Log.w("FileUtils", "Copying file data")
                input.copyTo(output)
                input.close()
                output.close()
                Log.w("FileUtils", "File saved successfully")
                true
            } else {
                Log.e("FileUtils", "Failed to open input or output stream")
                false
            }
        } catch (e: Exception) {
            Log.e("FileUtils", "Error saving file", e)
            false
        }
    }
    
    /**
     * 递归拷贝Assets目录到应用存储
     */
    fun copyAssetDirToFiles(context: Context, assetDir: String, outDir: File) {
        val am = context.assets
        val list = am.list(assetDir) ?: return
        
        if (!outDir.exists()) outDir.mkdirs()
        
        for (name in list) {
            val assetPath = if (assetDir.isEmpty()) name else "$assetDir/$name"
            val out = File(outDir, name)
            val children = am.list(assetPath)
            
            if (children != null && children.isNotEmpty()) {
                copyAssetDirToFiles(context, assetPath, out)
            } else {
                am.open(assetPath).use { input ->
                    FileOutputStream(out).use { output -> input.copyTo(output) }
                }
            }
        }
    }
    
    /**
     * 获取外部存储根目录
     */
    fun getExternalRoot(context: Context): File {
        return requireNotNull(context.getExternalFilesDir(null))
    }
    
    /**
     * 确保外部资源就绪
     */
    fun ensureExternalResourcesReady(context: Context): File {
        val root = getExternalRoot(context)
        val resDir = File(root, "resources")
        val marker = File(resDir, ".installed")
        if (!marker.exists()) {
            copyAssetDirToFiles(context, "resources", resDir)
            marker.parentFile?.mkdirs()
            marker.writeText(System.currentTimeMillis().toString())
        }
        return resDir
    }
}