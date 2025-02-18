
# model

nndeploy官方采用的模型文件放在modelscope社区和huggingface社区。

- nndeploy在modelscope社区的的官方仓库为[modelscope/nndeploy](https://www.modelscope.cn/models/nndeploy/nndeploy)
- nndeploy在huggingface社区的的官方仓库为[huggingface/nndeploy](https://huggingface.co/alwaysssss/nndeploy)

开发者完成模型部署后，需将对应的模型文件上传到以下平台并更新模型列表

- nndeploy在modelscope社区的的官方仓库[modelscope/nndeploy](https://www.modelscope.cn/models/nndeploy/nndeploy)
- nndeploy在huggingface社区的的官方仓库[huggingface/nndeploy](https://huggingface.co/alwaysssss/nndeploy)
- 开发者自己的modelscope社区仓库
- 开发者自己的huggingface社区仓库
- 开发者熟悉的开放平台

使用指南

- modelscope使用指南[modelscope使用指南](https://www.modelscope.cn/docs/home)
- huggingface使用指南[huggingface使用指南](https://huggingface.co/docs)

## 支持模型列表和下载链接

### classification

- [classification/resnet50-v1-7.onnx](./classification/resnet50-v1-7.onnx): ResNet50-v1-7, Model Type: onnx, input size: Nx3x224x224, classes: 1000, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/classification/resnet50-v1-7.onnx)
- [classification/resnet50-v1-7.sim.onnx](./classification/resnet50-v1-7.sim.onnx): onnx sim model of ResNet50-v1-7, Model Type: onnx, input size: Nx3x224x224, classes: 1000, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/classification/resnet50-v1-7.sim.onnx)
- [classification/resnet50-v1-7.slim.onnx](./classification/resnet50-v1-7.slim.onnx): onnx slim model of ResNet50-v1-7, Model Type: onnx, input size: Nx3x224x224, classes: 1000, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/classification/resnet50-v1-7.slim.onnx)
- [classification/resnet50-v1-7.staticshape.onnx](./classification/resnet50-v1-7.staticshape.onnx): static shape model of ResNet50-v1-7, Model Type: onnx, input size: 1x3x224x224, classes: 1000, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/classification/resnet50-v1-7.staticshape.onnx)
- [classification/resnet50-v1-7.staticshape.slim.onnx](./classification/resnet50-v1-7.staticshape.slim.onnx): static shape slim model of ResNet50-v1-7, Model Type: onnx, input size: 1x3x224x224, classes: 1000, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/classification/resnet50-v1-7.staticshape.slim.onnx)
- [classification/resnet50-v1-7.staticshape.onnx.json](./classification/resnet50-v1-7.staticshape.onnx.json)/[classification/resnet50-v1-7.staticshape.onnx.safetensor](./classification/resnet50-v1-7.staticshape.onnx.safetensor): static shape model of ResNet50-v1-7, Model Type: nndeploy, input size: 1x3x224x224, classes: 1000, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/classification/resnet50-v1-7.staticshape.onnx.json)
- [classification/resnet50-v1-7.onnx.om](./classification/resnet50-v1-7.onnx.om): ResNet50-v1-7, Model Type: AscendCL(Ascend910B4), input size: 1x3x224x224, classes: 1000, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/classification/resnet50-v1-7.onnx.om)

### detect

- [detect/yolov11s.onnx](./detect/yolov11s.onnx): YOLOv11s, Model Type: onnx, input size: Nx640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/detect/yolov11s.onnx)
- [detect/yolov11s.sim.onnx](./detect/yolov11s.sim.onnx): onnx sim model of YOLOv11s, Model Type: onnx, input size: 1x640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/detect/yolov11s.sim.onnx)
- [detect/yolov11s.slim.onnx](./detect/yolov11s.slim.onnx): onnx slim model of YOLOv11s, Model Type: onnx, input size: 1x640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/detect/yolov11s.slim.onnx)
- [detect/yolov11s.sim.onnx.json](./detect/yolov11s.sim.onnx.json)/[detect/yolov11s.sim.onnx.safetensor](./detect/yolov11s.sim.onnx.safetensor): YOLOv11s, Model Type: nndeploy, input size: 1x640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/detect/yolov11s.sim.onnx.json)
- [detect/yolov11s.onnx.om](./detect/yolov11s.onnx.om): YOLOv11s, Model Type: AscendCL(Ascend910B4), input size: 1x640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/detect/yolov11s.onnx.om)

- [detect/yolov8n.onnx](./detect/yolov8n.onnx): YOLOv8n, Model Type: onnx, input size: Nx640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/detect/yolov8n.onnx)

### segment

- [segment/RMBGV1.4.onnx](./segment/RMBGV1.4.onnx): RMBGV1.4, Model Type: onnx, input size: Nx1x1024x1024, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/segment/RMBGV1.4.onnx)
- [segment/RMBGV1.4.staticshape.onnx](./segment/RMBGV1.4.staticshape.onnx): static shape model of RMBGV1.4, Model Type: onnx, input size: 1x1x1024x1024, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/segment/RMBGV1.4.staticshape.onnx)
- [segment/RMBGV1.4.sim.onnx](./segment/RMBGV1.4.sim.onnx): onnx sim model of RMBGV1.4, Model Type: onnx, input size: 1x1x1024x1024, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/segment/RMBGV1.4.sim.onnx)
- [segment/RMBGV1.4.slim.onnx](./segment/RMBGV1.4.slim.onnx): onnx slim model of RMBGV1.4, Model Type: onnx, input size: 1x1x1024x1024, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/segment/RMBGV1.4.slim.onnx)
- [segment/RMBGV1.4.slim.onnx.json](./segment/RMBGV1.4.slim.onnx.json)/[segment/RMBGV1.4.slim.onnx.safetensor](./segment/RMBGV1.4.slim.onnx.safetensor): RMBGV1.4, Model Type: nndeploy, input size: 1x1x1024x1024, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/segment/RMBGV1.4.slim.onnx.json)

- [segment/RMBGV1.4.onnx.om](./segment/RMBGV1.4.onnx.om): RMBGV1.4, Model Type: AscendCL(Ascend910B4), input size: 1x1x1024x1024, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/segment/RMBGV1.4.onnx.om)



