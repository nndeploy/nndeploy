# segment anything 

## 算法
+ github
  + https://github.com/facebookresearch/segment-anything
+ demo
  + https://segment-anything.com/
+ HuggingFace
  + facebook/sam-vit-huge: https://huggingface.co/facebook/sam-vit-huge
  + facebook/sam-vit-large https://huggingface.co/facebook/sam-vit-large
  + facebook/sam-vit-base https://huggingface.co/facebook/sam-vit-base

## 模型
采用sam-vit-base该较小的模型
+ 通HuggingFace下载模型
  ```
  git clone git@hf.co:facebook/sam-vit-base
  ```
  + 文件分析
    + pytorch_model.bin - pytorch训练得到的权重文件，不好包模型结构，无法用netron打开
    + tf_model.h5 - tensorflow训练得到的权重文件，不包含模型结构，无法用netron打开
    + config.json - 模型结构的配置文件
    + preprocessor_config.json - 模型前处理配置文件
    + README.md - 仓库解释，包含了基于transfomers的推理代码
+ 基于transfomers的推理 + 导出torchscript模型 + 导出onnx模型文件
  + resourcesegment_anything.py
+ 模型信息查询
  + 
## 模型修改[optional]

## 模型图优化

## 模型python推理

## 模型转换

## 模型c++推理

## demo