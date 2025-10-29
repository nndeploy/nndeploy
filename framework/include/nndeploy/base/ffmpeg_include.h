#ifndef _NNDEPLOY_BASE_FFMPEG_INCLUDE_H_
#define _NNDEPLOY_BASE_FFMPEG_INCLUDE_H_

#define ENABLE_NNDEPLOY_FFMPEG

#ifdef ENABLE_NNDEPLOY_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#endif

#endif