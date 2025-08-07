# nndeploy Docker Image (Ubuntu 22.04)

This Docker image builds and runs the [nndeploy](https://github.com/nndeploy/nndeploy) framework in a clean Ubuntu 22.04 environment. It includes all dependencies such as OpenCV, ONNX Runtime, and builds both the C++ backend and Python package.

## Available Docker Images

We currently support the following Docker configurations:

- **Dockerfile**: Basic version with ONNX Runtime only
- **Dockerfile.ort_ov**: Includes ONNX Runtime, OpenVINO
- **Dockerfile.ort_ov_trt**: Includes ONNX Runtime, OpenVINO, TensorRT
- **Dockerfile.ort_ov_mnn_trt**: Includes ONNX Runtime, OpenVINO, TensorRT, MNN
<!-- - **Dockerfile.ort_ascend**: Includes ONNX Runtime and Huawei Ascend inference engine
- **Dockerfile.ort_rknn**: Includes ONNX Runtime and Rockchip RKNN inference engine -->

The following instructions use `Dockerfile` as an example, but the same operations apply to all other Docker files.

## Build the Image

```bash
docker build -f docker/Dockerfile -t nndeploy-linux .
````

## Run the Container (Default Port 8888)

```bash
docker run -it -p 8888:8888 nndeploy-linux
```

This will run:

```bash
python3 app.py --port 8888
```

## Run with Custom Port

```bash
docker run -it -p 9000:9000 nndeploy-linux python3 app.py --port 9000
```

## Run with Shell

```bash
docker run -it nndeploy-linux bash
```

## Save and Share the Image

```bash
# Save image
docker save nndeploy-linux -o nndeploy-linux.tar

# On another machine
docker load -i nndeploy-linux.tar
```

## Notes

* `.so` files are located at `/workspace/python/nndeploy` and are registered in `ldconfig`.
* You can modify `app.py` as needed before building.
* If you change source code, rebuild the image.