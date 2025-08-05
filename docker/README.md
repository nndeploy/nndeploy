# nndeploy Docker Image (Ubuntu 22.04)

This Docker image builds and runs the [nndeploy](https://github.com/nndeploy/nndeploy) framework in a clean Ubuntu 22.04 environment. It includes all dependencies such as OpenCV, ONNX Runtime, and builds both the C++ backend and Python package.

## Build the Image

```bash
docker build -f docker/Dockerfile -t nndeploy-linux .
````

## Run the Container (Default Port 8000)

```bash
docker run -it -p 8000:8000 nndeploy-linux
```

This will run:

```bash
python3 app.py --port 8000
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