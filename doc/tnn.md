# tnn
## 目标
+ 以yolov5例程，跑通tnn
+ 能够使用tnn推理接口，并理解tnn推理接口含义
+ 能够使用tnn转换器，并理解其接口背后含义
## 工作任务分解
+ 走读tnn上层目录（最顶目录和次级目录），理解各个目录背后的含义
+ 编译tnn库、tnn-yolov5例程、tnn转换器
  + 阅读tnn主目录下的cmakelist文件
  + 在tnn主目录下建立build目录，用于存放所有cmake以及build生成的文件
  + 根据cmakelist去写cmake命令行，cmake 以及 make命令生成tnn库以及tnn-yolov5可执行程序以及tnn转换器库
+ 跑通yolov5例程
  + 下载yolov5模型文件
  + 根据tnn-yolov5可执行程序传入模型文件以及图片得到运行结果
+ 理解yolov5例程 
  + 可以使用vs自带的调试工具，进行逐步调试
  + 源码的流程是 main（examples\linux\src\TNNObjectDetector\TNNObjectDetector.cc）、yolov5的具体库（examples\base\object_detector_yolo.h, C:\github\TNN\examples\base\tnn_sdk_sample.h）、tnn（C:\github\TNN\include\tnn）
  + 阅读C:\github\TNN\examples\linux\src\TNNObjectDetector\TNNObjectDetector.cc可执行源码，理解其含义，有部分接口不用去关注他的细节，我会告诉你他的作用，比如ParseAndCheckCommandLine，其主要作用是解析你传入命令行参数；需要理解包括如下
    + 如何初始化yolo
    + 图片读入
    + 如何做前处理
    + 如何做推理
    + 如何做后处理
    + 如何处理后处理的框
    + 如何释放各类资源
+ 阅读examples\base\object_detector_yolo.h, C:\github\TNN\examples\base\tnn_sdk_sample.h源码，理解其含义
  + 如何初始化
  + 如何做前处理
  + 如何推理
  + 如何做后处理
  + 如何释放资源
+ 阅读C:\github\TNN\include\tnn
  + 如何做初始化
  + 如何做推理
  + 如何释放资源
+ 能够使用tnn转换器，并理解其接口背后含义
  + 去hugging face下载一个模型，并把该模型转换onnx模型文件格式，使用netron查看模型文件
  + 使用onnxxoptimizer 和 onnxsim优化该模型，使用netron查看模型文件
  + 使用tnn的转化器转换该模型文onnx的模型接口，使用netron查看模型文件
  + 阅读C:\github\TNN\tools\converter\source\tnn_converter.cc源码。可以加固理解onnx的模型结构以及c++