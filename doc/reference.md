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
  + nndeploy
    + 场景分割（NDHWC五维模型）
      + openvino 270ms
      + onnxruntime 350ms
      + nndeploy 970ms
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
    + cv::parallel_for多线程的设计方法
  + ptmalloc
    + 传统内存池

+ clang-format
+ clang-tidy
  + 参考 https://www.inktea.eu.org/2022/b47a.html
  + 参考 https://github.com/llvm-mirror/clang-tools-extra
+ git commit 
  + 参考文档 https://developer.aliyun.com/article/929807
+ 注释
  + 插件 Doxygen Documentation Generator 不要文件头注释 全部采用该工具默认配置      
+ 文档
  + 基本规则 https://www.jianshu.com/p/ebe52d2d468f
  + 参考写法 https://github.com/openmlsys/openmlsys-zh
+ 代码风格
  + 不确定的地方建议阅读 https://zh-google-styleguide.readthedocs.io/en/latest/
  + 不想阅读文档可以参看 nndeploy 的实现方式
  + 其他任何语言代码风格 一律参考 nndeploy