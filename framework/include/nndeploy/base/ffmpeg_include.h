#ifndef _NNDEPLOY_BASE_FFMPEG_INCLUDE_H_
#define _NNDEPLOY_BASE_FFMPEG_INCLUDE_H_

#ifdef ENABLE_NNDEPLOY_FFMPEG
extern "C" {
    #include <libavformat/avformat.h>  // 处理容器格式，打开文件、流等
    #include <libavcodec/avcodec.h>    // 解码/编码音视频流
    #include <libavutil/avutil.h>      // 工具函数、数据类型等
    #include <libswscale/swscale.h>    // 图像格式转换与缩放
    #include <libswresample/swresample.h> // 音频重采样
    #include <libavutil/frame.h> // 图像处理工具函数
    #include <libavformat/avformat.h>
}
#endif

#endif /* _NNDEPLOY_BASE_FFMPEG_INCLUDE_H_ */