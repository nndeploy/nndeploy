// /home/always/github/public/nndeploy/app/android/app/src/main/java/com/nndeploy/base/VideoUtils.kt
package com.nndeploy.base

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import java.io.File

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
        onVideoRecorded: (Uri?) -> Unit
    ): ActivityResultLauncher<Uri> {
        return activity.registerForActivityResult(ActivityResultContracts.CaptureVideo()) { success ->
            if (success) {
                // 这里需要传入预设的 URI
                onVideoRecorded(null) // 实际实现需要完善
            } else {
                onVideoRecorded(null)
            }
        }
    }
}