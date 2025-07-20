
include(ExternalProject)

if (ENABLE_NNDEPLOY_FFMPEG STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_FFMPEG STREQUAL "ON")
  set(FFMPEG_PATH /home/jodio/code/FFmpeg/ffmpeg_build/)
  # set(ENV{PKG_CONFIG_PATH} "third_party/ffmpeg/install/lib/pkgconfig")
  # find_package(PkgConfig REQUIRED)
  # pkg_check_modules(AVCODEC REQUIRED libavcodec)
  # pkg_check_modules(AVFORMAT REQUIRED libavformat)
  # pkg_check_modules(AVUTIL REQUIRED libavutil)
  # pkg_check_modules(AVFILTER REQUIRED libavfilter)
  # pkg_check_modules(AVDEVICE REQUIRED libavdevice)
  # pkg_check_modules(SWRESAMPLE REQUIRED libswresample)
  # pkg_check_modules(SWSCALE REQUIRED libswscale)
  # include_directories(${AVCODEC_INCLUDE_DIRS} ${AVFORMAT_INCLUDE_DIRS} ${AVUTIL_INCLUDE_DIRS}
  #                     ${AVFILTER_INCLUDE_DIRS} ${AVDEVICE_INCLUDE_DIRS} ${SWRESAMPLE_INCLUDE_DIRS}
  #                     ${SWSCALE_INCLUDE_DIRS})
  include_directories(${FFMPEG_PATH}/include)
  set(AVCODEC_LIBRARIES ${FFMPEG_PATH}/lib/libavcodec.so)
  set(AVFORMAT_LIBRARIES ${FFMPEG_PATH}/lib/libavformat.so)
  set(AVUTIL_LIBRARIES ${FFMPEG_PATH}/lib/libavutil.so)
  set(AVFILTER_LIBRARIES ${FFMPEG_PATH}/lib/libavfilter.so)
  set(AVDEVICE_LIBRARIES ${FFMPEG_PATH}/lib/libavdevice.so)
  set(SWRESAMPLE_LIBRARIES ${FFMPEG_PATH}/lib/libswresample.so)
  set(SWSCALE_LIBRARIES ${FFMPEG_PATH}/lib/libswscale.so)
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} 
                                   ${AVCODEC_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVUTIL_LIBRARIES}
                                   ${AVFILTER_LIBRARIES} ${AVDEVICE_LIBRARIES} ${SWRESAMPLE_LIBRARIES}
                                   ${SWSCALE_LIBRARIES})
  add_definitions(-DENABLE_NNDEPLOY_FFMPEG)

else()
  include_directories(${ENABLE_NNDEPLOY_FFMPEG}/include)
  set(LIB_PATH ${ENABLE_NNDEPLOY_FFMPEG}/${NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX})
  foreach(LIB ${NNDEPLOY_OPENCV_LIBS})
    set(LIB_NAME ${NNDEPLOY_LIB_PREFIX}${LIB}${NNDEPLOY_LIB_SUFFIX}${NNDEPLOY_OPENCV_VERSION})
    set(FULL_LIB_NAME ${LIB_PATH}/${LIB_NAME})
    set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${FULL_LIB_NAME})    
  endforeach()
  install(DIRECTORY ${ENABLE_NNDEPLOY_FFMPEG}
          DESTINATION ${NNDEPLOY_INSTALL_THIRD_PARTY_PATH}
          PATTERN "opencv" EXCLUDE)
endif()