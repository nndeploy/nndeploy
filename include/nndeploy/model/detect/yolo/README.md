# yolo

## yolov6
+ github
  + https://github.com/meituan/YOLOv6
+ 2022美团技术总结
  + https://s3plus.meituan.net/v1/mss_e63d09aec75b41879dcb3069234793ac/file/2022%E5%B9%B4%E7%BE%8E%E5%9B%A2%E6%8A%80%E6%9C%AF%E5%B9%B4%E8%B4%A7-%E5%90%88%E8%BE%91.pdf

## 模型
+ 通官方页面 https://github.com/meituan/YOLOv6 下载模型
+ 导出onnx模型文件
  + python ./deploy/ONNX/export_onnx.py --weights /home/always/github/public/nndeploy/resourcemodel/yolo/yolov6l.pt --simplify
  + python ./deploy/ONNX/export_onnx.py --weights /home/always/github/public/nndeploy/resourcemodel/yolo/yolov6s.pt --simplify
    + 运行失败
  + python ./deploy/ONNX/export_onnx.py --weights /home/always/github/public/nndeploy/resourcemodel/yolo/yolov6m.pt --simplify
  + python ./deploy/ONNX/export_onnx.py --weights /home/always/github/public/nndeploy/resourcemodel/yolo/yolov6n.pt --simplify
    + 运行失败