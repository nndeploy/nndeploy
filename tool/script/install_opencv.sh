# 下载源码（以 4.8.0 为例）
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip
unzip opencv.zip
cd opencv-4.8.0
mkdir build && cd build

# 编译安装
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-4.8.0 ..
make -j$(nproc)
sudo make install

# 配置环境变量
echo 'export LD_LIBRARY_PATH=/usr/local/opencv-4.8.0/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc