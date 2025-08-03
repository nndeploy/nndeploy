# docker

## 构建镜像
docker build -f docker/Dockerfile.ort -t nndeploy:ubuntu22.04 .

docker build \
  --build-arg HTTP_PROXY=http://127.0.0.1:7890 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:7890 \
  -f docker/Dockerfile.ort \
  -t nndeploy:ubuntu22.04 .

## 运行容器

# 基本运行
docker run --rm nndeploy:ubuntu22.04

# 交互模式
docker run -it --rm nndeploy:ubuntu22.04 /bin/bash

# 开发模式（挂载本地代码）
docker run -it --rm \
  -v $(pwd):/workspace/nndeploy \
  --workdir /workspace/nndeploy/build \
  nndeploy:ubuntu22.04 /bin/bash