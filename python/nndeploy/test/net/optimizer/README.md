# Optimizer测试

该文件夹用于图优化中各pass的测试。对于每一个pass的测试，均按照以下流程：

1. 手动搭建网络，该网络的架构经过精心设计，为实现该图优化的最小结构

2. 使用相同的权重和计算逻辑，实现PyTorch的计算代码，作为基准Baseline

3. 提供多种图优化搭配的选项，并进行结果对比

## test_eliminate_common_subexpression

测试消除公共子表达式的pass。

运行代码

```python
cd nndeploy/python/nndeploy/test/net/optimizer
python test_eliminate_common_subexpression.py

```

## test_eliminate_dead_op

测试消除死节点的pass。

运行代码

```python
cd nndeploy/python/nndeploy/test/net/optimizer
python test_eliminate_dead_op.py

```

## test_fuse_conv_relu

测试conv+relu算子融合的pass。

运行代码

```python
cd nndeploy/python/nndeploy/test/net/optimizer
python test_fuse_conv_relu.py

```

