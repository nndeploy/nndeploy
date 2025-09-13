// /home/always/github/public/nndeploy/app/android/app/src/main/java/com/nndeploy/base/camerautils.kt
package com.nndeploy.base

import android.content.Context
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.ComponentActivity
import android.net.Uri

/**
 * 摄像头和图片选择工具类
 */
object CameraUtilsUtils {
    /**
     * 创建相机拍照启动器
     */
    fun createCameraUtilsLauncher(
        activity: ComponentActivity,
        onPhotoTaken: (Uri?) -> Unit
    ): ActivityResultLauncher<Uri> {
        return activity.registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
            if (success) {
                // 这里需要传入预设的 URI
                onPhotoTaken(null) // 实际实现需要完善
            } else {
                onPhotoTaken(null)
            }
        }
    }
}