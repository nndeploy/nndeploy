#!/bin/bash

chmod +x Ascend-cann-toolkit_8.0.RC3.alpha003_linux-aarch64.run
./Ascend-cann-toolkit_8.0.RC3.alpha003_linux-aarch64.run --check
./Ascend-cann-toolkit_8.0.RC3.alpha003_linux-aarch64.run --install
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/devlib/:$LD_LIBRARY_PATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc

chmod +x Ascend-cann-kernels-910b_8.0.RC3.alpha003_linux-aarch64.run
./Ascend-cann-kernels-910b_8.0.RC3.alpha003_linux-aarch64.run --check
./Ascend-cann-kernels-910b_8.0.RC3.alpha003_linux-aarch64.run --install

if [ -d "cmake-3.28.3-linux-aarch64" ]; then
    echo "The cmake folder exists, skipping the download and decompression steps."
elif [ -e "cmake-3.28.3-linux-aarch64.tar.gz" ]; then
    echo "CMake compressed file exists, start decompressing steps."
    tar xf cmake-3.28.3-linux-aarch64.tar.gz
else
    echo "need CMake compressed file exists."
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/AscendC/ResourceDependent/cmake-3.28.3-linux-aarch64.tar.gz
    tar xf cmake-3.28.3-linux-aarch64.tar.gz
fi
export PATH=/usr/local/work/cmake-3.28.3-linux-aarch64/bin:$PATH
echo 'export PATH=/usr/local/work/cmake-3.28.3-linux-aarch64/bin:$PATH' >> ~/.bashrc

