// /home/always/github/public/nndeploy/app/android/app/src/main/java/com/nndeploy/base/ImageUtils.kt
package com.nndeploy.base

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
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
}