# 社区问题

## 问题一 来自 nndeploy交流群
### Q: 我想了解一下：onnx runtime也可以选择不同的execution provider: https://onnxruntime.ai/docs/execution-providers/ nndeploy与其思路的差别主要在哪里呢？ 
### A: ONNXRuntime与nndeploy最大的区别是：ONNXRuntime首先是一个的推理框架，然后可以接入其他推理框架，nndeploy明确定位就是一个多端部署框架。这个会带来的差异主要是如下两点
- ONNXRuntime为了保证其他推理框架兼容其自身，在功能上会对其他推理框架有一定程度的阉割（比如TNN、mnn操作推理框架内部分配的输入输出、OpenVINO CPU-GPU异构模式、模型量化等等），导致模型通过EP只是能跑起来，性能应该不好。nndeploy希望类似：你搞了很久的trt，然后你就trt搭建了一个自己的小型框架，可以完全操纵trt;
- ONNXRuntime强绑定ONNX以及protobuf，会导致包体积变大，尤其是对于移动端不友好，对于不能直接读取onnx模型文件的框架应该也不太友好，还有就是现在很多框架都是pt->自定义模型文件（例如ncnnn\mnn），以达到极致的性能

总结而言，都是调用第三方推理框架，使用nndeploy功能更全面、性能更好、体验更丝滑。

## 问题二 来自 nndeploy交流群
### Q: nndeploy相对于openmmlab的mmdeploy以及paddle的fastdeploy会有什么优势？
### A: 非常开心可以和openmmlab的mmdeploy以及paddle的fastdeploy放在一起比较，毫无疑问，openmmlab的mmdeploy以及paddle的fastdeploy都是行业巨擘，有非常大影响力，据我所知都有非常多公司在使用这个两个库。因为我们是nndeploy的开发者，所以我们这里主要谈谈我们的优势。
- 更加开放。目前fd家重点是paddle的模型仓库以及主推自家推理框架，mmdp家主要是openmmlab的模型仓库以及主推自家推理框架，而nndeploy积极集成各家的推理框架，不设限的部署热门的开源模型。nndeploy想跳出这个"特定优先支持"，做一个更加通用普适的部署框架
- 更加易用，在多端推理的基础功能的基础上nndeploy还做了设备管理，让你可以有统一的方式操作内存以及执行流等；还通过有向无环图来管理模型部署的前处理、推理、后处理
- 更加适合部署多模型，nndeploy以有向无环图的形式来实现多个模型的部署，通过这种方式来部署多模型算法，可以少写大量的业务代码，模块性以及鲁棒性都会更好。
- 推理框架的高性能抽象：每个推理框架也都有其各自的特性，需要足够尊重以及理解这些推理框架，才能在抽象中不丢失推理框架的特性，并做到统一的使用的体验。nndeploy可配置第三方推理框架绝大部分参数，保证了推理性能。可直接操作理框架内部分配的输入输出，实现前后处理的零拷贝，提升模型部署端到端的性能。我们正在实现更多性能方面的优化，比如在图的基础上加上线程池、内存池、高性能算子库等，未来性能方面的功能会逐渐更加完善
- nndeploy更加专注模型c++部署，更轻量化一些
