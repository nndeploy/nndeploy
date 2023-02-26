# reference
  + tnn
    + 整体框架
  + openvino
    + qnnpack
    + cpu-gpu异构执行
    + 量化
  + tf-lte
    + xnnpack
  + mnn
    + 子图的拆分
    + flatbuffer
  + onnxruntime
    + python接口
    + 不采用基于图的内存分配，内存性能最好
  + tvm
    + 场景分割（NDHWC五维模型）
      + openvino 270ms
      + onnxruntime 350ms
      + tvm 970ms
    + 推理框架目前性能远远好于机器学习编译框架
  + trt
    + 最好写的自定义算子插件
  + tengine
    + 对推理框架完整的认识
  + aitemplate
    + 假如有可能的化，像借鉴它的整体实现思路
  + onnx
    + ir的设计，基于onnx ir, 但是增加更多优化的空间
  + onnx-optimizer
    + 基于onnx ir做图优化
  + onnx-sim
    + 基于onnxruntime做常量传播、常量折叠
  + ppl.cv
    + 高性能算子
  + opencv
    + 接口的设计
    + cv::Mat、cv::GpuMat、cv::InputArray、cv::OutputArray的设计
    + cv::Parallel_for多线程的设计方法
  + ptmalloc
    + 传统内存池