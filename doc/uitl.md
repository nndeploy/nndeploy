# util
## linux命令行
+ chown -R always /home/always/github/public/nndeploy/resourcemodel
+ chgrp -R nndeploy /home/always/github/public/nndeploy/resourcemodel
+ export LD_LIBRARY_PATH=/home/always/github/public/nndeploy/build/install:$LD_LIBRARY_PATH
+ export PATH=/usr/local/cuda/bin:$PATH
+ chgrp -R nndeploy /home/always/huggingface/nndeploy
+ chown -R always /home/always/huggingface/nndeploy

## cmake 
+ cmake .. -DCMAKE_BUILD_TYPE=Debug
+ cmake .. -DCMAKE_INSTALL_PREFIX=C:\nndeploy\build\install
+ 
+ cmake -DCMAKE_TOOLCHAIN_FILE=${NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_STL="c++_static" -DANDROID_NATIVE_API_LEVEL="android-14" -DANDROID_TOOLCHAIN=clang -DBUILD_FOR_ANDROID_COMMAND=true ..
+ cmake -DCMAKE_TOOLCHAIN_FILE=${NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_STL="c++_static" -DANDROID_NATIVE_API_LEVEL="android-14" -DANDROID_TOOLCHAIN=clang -DBUILD_FOR_ANDROID_COMMAND=true ..

