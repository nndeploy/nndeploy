FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ---- Install System Deps ----
RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build git wget curl pkg-config \
    python3-dev python3-pip python3-venv \
    libopencv-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python Libraray ----
RUN python3 -m pip install --upgrade pip
RUN pip install pybind11 setuptools wheel cython requests pathlib

# ---- Install Rust（Optional）----
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# ---- Clone nndeploy Project（Include Submodules）----
RUN git clone --recursive https://github.com/nndeploy/nndeploy.git .

# ---- Install OpenCV + ONNX Runtime ----
RUN cd tool/script && \
    python3 install_opencv.py && \
    python3 install_onnxruntime.py && \
    cd ../..

# ---- Build C++ and Install Python Module ----
RUN rm -rf build && \
    mkdir build && \
    cp cmake/config_opencv_ort_tokenizer.cmake build/config.cmake && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. && \
    make -j$(nproc) && \
    make install

# ---- Install Python Package ----
RUN cd python && \
    python3 setup.py bdist_wheel && \
    pip install dist/nndeploy-0.2.10-cp310-cp310-linux_x86_64.whl && \
    cd ..

# ---- Default Interface ----
WORKDIR /workspace/
CMD ["python3", "app.py"]
