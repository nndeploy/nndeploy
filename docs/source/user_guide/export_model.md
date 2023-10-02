
# 导出模型
将训练框架模型导出为onnx模型文件，有两种方式，直接导出和通过训练框架模型文件导出

## 直接导出，调用训练框架接口直接导出为onnx模型文件

## 通过训练框架模型文件导出
### pytorch模型文件类型
有多少中模型保存的格式呢？分别都有什么特点呢？
模型文件扩展名
+ .pt、.pth
+ .bin (HuggingFace模型通常都为该格式)

#### state_dict模型
+ 使用方法
  ```
  # 保存
  torch.save(model.state_dict(), PATH)
  # 加载
  # 模型类必须在此之前被定义
  model.load_state_dict(torch.load(PATH))
  model.eval()
  ```
+ 特点
  + 通过state_dict的Python字典对象，它将模型中每一层训练参数通过[key、value]形式保存，为模型训练提供最大的灵活性
  + 未保留模型结构，故一定需要和nn.Module相关的对象绑定才可以使用
  + 不可以直接转换为其他推理框架的模型文件

#### checkpoint模型 
+ 使用方法
  ```
  # 保存
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
  # 加载
  # 模型类必须在此之前被定义
  checkpoint = torch.load(PATH)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  model.eval()
  model.train()
  ```
+ 特点
  + 有state_dict模型全部特点
  + 主要是保存模型训练时的中间模型，故该种模型会保存很多训练相关的信息，例如优化器、epoch等

#### 完整模型
+ 使用方法
  ```
  # 保存
  torch.save(model, PATH)
  # 加载
  model = torch.load(PATH)
  model.eval()
  ```
+ 特点
  + 保留了模型结构+模型参数全部信息
  + 对model该类的修改，可能会导致项目中断，不利于项目重构、微调等
  
#### 注意
+ 在运行推理之前，务必调用model.eval()设置 dropout 和 batch normalization 层为评估模式。

### 导出为onnx

### onnxsim