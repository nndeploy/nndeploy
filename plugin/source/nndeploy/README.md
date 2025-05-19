
# plugin

+ 所有算法的前处理放在`plugin/source/nndeploy/preprocess`

+ 每类算法需要定义一个Result数据结构，后处理跟具体算法关联

## 阅读源码

+ zuiren：fastdeploy
+ always：mmdeploy

+ 直接使用fastdeploy和mmdeploy的模型，把模型相关的链接放到demo/xxx/readme.md中

## 分工

+ zuiren
  + classification
  + detect(包含人脸检测)
  + segmentation（或许需要包含matting）
  + track（暂时不做传统的track）
  + llm
  + stable diffusition
+ always 
  + face_id
  + face_align
  + gan
  + ocr
  + repair
  + super_resolution