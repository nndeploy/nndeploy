# yolo

## yolov3
+ github
  + https://github.com/ultralytics/yolov3

## 模型
+ 通官方页面 https://github.com/meituan/YOLOv6 下载模型
+ 导出onnx模型文件
  + python ./deploy/ONNX/export_onnx.py --weights /home/always/github/public/nndeploy/resourcemodel/yolo/yolov6l.pt --simplify
  + python ./deploy/ONNX/export_onnx.py --weights /home/always/github/public/nndeploy/resourcemodel/yolo/yolov6s.pt --simplify
    + 运行失败
  + python ./deploy/ONNX/export_onnx.py --weights /home/always/github/public/nndeploy/resourcemodel/yolo/yolov6m.pt --simplify
  + python ./deploy/ONNX/export_onnx.py --weights /home/always/github/public/nndeploy/resourcemodel/yolo/yolov6n.pt --simplify
    + 运行失败