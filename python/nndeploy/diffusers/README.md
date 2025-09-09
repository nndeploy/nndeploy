
# 

## 无法下载hugging face的模型

+ 参考huging face mirror: https://hf-mirror.com/
  + 快捷命令：
    + export HF_ENDPOINT=https://hf-mirror.com
    + $env:HF_ENDPOINT = "https://hf-mirror.com"

+ LOADABLE_CLASSES是啥？ 
  + zh: 

+ diffusers的pipeline保存后，其目录结构为
```bash
save_directory/
├── model_index.json          # 管道配置
├── unet/                     # UNet模型
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vae/                      # VAE模型
├── scheduler/                # 调度器
├── text_encoder/            # 文本编码器
└── tokenizer/               # 分词器
```bash

+ def to(self, *args, **kwargs) -> Self:
  + 这是DiffusionPipeline的设备转移方法，用于将整个管道移动到指定设备
  + 参数说明：
    + *args: 位置参数，通常是设备标识符
      + 可以是字符串形式："cuda", "cpu", "cuda:0", "cuda:1" 等
      + 可以是torch.device对象：torch.device("cuda:0")
      + 可以是torch的数据类型：torch.float16, torch.float32 等
    + **kwargs: 关键字参数，用于更精细的控制
      + device: 明确指定目标设备
      + dtype: 指定数据类型转换
      + non_blocking: 是否使用非阻塞传输（默认False）
      + memory_format: 内存格式（通常不需要指定）
  + 使用示例：
    + pipeline.to("cuda")  # 移动到GPU
    + pipeline.to("cpu")   # 移动到CPU  
    + pipeline.to("cuda:1") # 移动到第二块GPU
    + pipeline.to(torch.float16) # 转换数据类型
    + pipeline.to("cuda", dtype=torch.float16) # 同时指定设备和数据类型
  + 注意事项：
    + 该方法会移动管道中的所有组件（UNet、VAE、文本编码器等）
    + 返回Self类型，支持链式调用
    + 大模型转移可能需要较长时间和足够的显存

+ 什么时dduf模型文件

+ 镜像的使用方法：示例：`DiffusionPipeline.from_pretrained("model_name", mirror="https://hf-mirror.com")`

+ 设备映射策略，用于指定管道的不同组件应如何分布在可用设备上。这对于处理大型模型特别有用，
                可以将模型的不同部分分配到不同的GPU或CPU上以优化内存使用。目前仅支持 "balanced" 设备映射策略，
                该策略会自动平衡地将模型组件分配到可用设备上。更多信息请参考
                [此文档](https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement)。
                
                示例用法：
                ```python
                # 使用平衡设备映射策略加载大型管道
                from diffusers import DiffusionPipeline
                
                pipeline = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    device_map="balanced"  # 自动平衡分配组件到可用设备
                )
                
                # 这将自动将UNet、VAE、文本编码器等组件分配到不同的GPU上
                # 以优化内存使用并避免OOM错误

+ max_memory (`Dict`, *optional*):
                一个设备标识符到最大内存的字典映射。用于指定每个设备（如GPU、CPU）可使用的最大内存量。
                如果未设置，将默认使用每个GPU的最大可用内存以及可用的CPU内存。
                
                使用方法：
                - 字典的键应为设备标识符：使用 "0", "1", "2" 等表示GPU设备编号，使用 "cpu" 表示CPU设备
                - 字典的值为内存大小：可以是整数（以字节为单位）或字符串格式（如 "1GB", "500MB", "2GiB"）
                - 示例：{"0": "8GB", "1": "8GB", "cpu": "30GB"} 表示GPU 0和GPU 1各使用最多8GB内存，CPU使用最多30GB内存
                - 示例：{"0": 8000000000, "cpu": "16GB"} 表示GPU 0使用最多8GB内存（以字节为单位），CPU使用最多16GB内存
                
                这对于在内存受限的环境中控制模型加载和推理时的内存使用非常有用，可以避免内存溢出错误。

+ offload_folder (`str` or `os.PathLike`, *optional*):
                指定权重卸载到磁盘的路径。当 `device_map` 参数包含 `"disk"` 值时，模型的某些组件会被卸载到指定的磁盘目录中以节省内存。
                如果未指定，将使用默认的临时目录。这对于处理大型模型时非常有用，可以避免内存不足的问题。
                
                示例：
                ```python
                # 将权重卸载到指定目录
                pipeline = DiffusionPipeline.from_pretrained(
                    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    device_map={"unet": "disk", "vae": 0},
                    offload_folder="./model_offload"
                )
                
                # 使用默认临时目录进行卸载
                pipeline = DiffusionPipeline.from_pretrained(
                    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    device_map={"text_encoder": "disk"}
                )
                ```

+ offload_state_dict (`bool`, *optional*):
                如果设置为 `True`，会临时将CPU状态字典卸载到硬盘上，以避免在CPU状态字典的权重加上检查点最大分片的大小超出内存时导致CPU RAM不足。当存在磁盘卸载时，默认为 `True`。
                
                这个参数主要用于处理大型模型加载时的内存管理问题。当模型权重文件很大，而系统内存有限时，通过将状态字典临时存储到磁盘可以避免内存溢出。
                
                示例场景：
                - 加载一个30GB的模型，但系统只有16GB RAM时
                - 在资源受限的环境中加载多个大型模型组件时
                - 使用device_map="disk"进行磁盘卸载时会自动启用此功能

+ 详解：pickle和safetensors等文件，如何与torch.nn.Module关联起来的呢

+ 激活值

+ export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

+ 详解torch中tensor，从python侧到最底层的cpp侧

+ 什么是xla