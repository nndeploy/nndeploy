# util
## linux命令行
+ chown -R always /home/always/github/public/nndeploy/resourcemodel
+ chgrp -R nndeploy /home/always/github/public/nndeploy/resourcemodel
+ export LD_LIBRARY_PATH=/home/always/github/public/nndeploy/build/install:$LD_LIBRARY_PATH
+ export PATH=/usr/local/cuda/bin:$PATH
+ chgrp -R nndeploy /data/local/tmp
+ chown -R always /data/local/tmp

## cmake 
+ cmake .. -DCMAKE_BUILD_TYPE=Debug
+ cmake .. -DCMAKE_INSTALL_PREFIX=C:\nndeploy\build\install
+ 
+ cmake -DCMAKE_TOOLCHAIN_FILE=${NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_STL="c++_static" -DANDROID_NATIVE_API_LEVEL="android-14" -DANDROID_TOOLCHAIN=clang -DBUILD_FOR_ANDROID_COMMAND=true ..
+ cmake -DCMAKE_TOOLCHAIN_FILE=${NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_STL="c++_static" -DANDROID_NATIVE_API_LEVEL="android-14" -DANDROID_TOOLCHAIN=clang -DBUILD_FOR_ANDROID_COMMAND=true ..
+ export LD_LIBRARY_PATH=/data/local/tmp/lib:$LD_LIBRARY_PATH
+ adb push lib/ /data/local/tmp
+ adb push yolov6m.onnx.mnn /data/local/tmp/model_zoo
+ adb push yolov6m.tnn* /data/local/tmp/model_zoo
+ adb push temp/ /data/local/tmp/
+ adb push test_data/ /data/local/tmp/
+ adb pull /data/local/tmp/temp /home/always/huggingface/nndeploy/temp

./demo_nndeploy_detect --name YOLO_NAME --inference_type kInferenceTypeTnn --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeTnn --is_path --model_value /data/local/tmp/model_zoo/yolov6m.tnnproto,/data/local/tmp/model_zoo/yolov6m.tnnmdodel --input_type kInputTypeImage  --input_path /data/local/tmp/test_data/detect/sample.jpg --output_path /data/local/tmp/temp/sample_output_and.jpg


./demo_nndeploy_detect --name YOLO_NAME --inference_type kInferenceTypeMnn --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeMnn --is_path --model_value /data/local/tmp/model_zoo/yolov6m.onnx.mnn --input_type kInputTypeImage  --input_path /data/local/tmp/test_data/detect/sample.jpg --output_path /data/local/tmp/temp/sample_output_and.jpg