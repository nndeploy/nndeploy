
# Diffusion Pipeline 快速指南

本手册简明介绍如何在 nndeploy 中部署与使用 text2image、image2image、inpainting。

---

## 1. 模型下载与环境配置

- **国内加速**：  
  启动前设置 hugging face 镜像，加速模型下载。  
  Linux/macOS:
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
  Windows PowerShell:
  ```powershell
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  ```

- **受限模型访问**：  
  需在 Hugging Face 申请权限、获取 Access Token 并登录。  
  1. 访问模型页面，申请权限  
  2. [获取 Access Token](https://huggingface.co/settings/tokens)  
  3. 安装工具并登录：
     ```
     pip install huggingface_hub
     huggingface-cli login --token <your_token>
     ```

---

## 2. 内存与性能优化

- **enable_model_cpu_offload**：开启后大幅降低显存占用，推荐显存不足时使用。
- **enable_sequential_cpu_offload**：进一步降低显存占用，速度略降。
- **xformers**：安装后开启 `enable_xformers_memory_efficient_attention`，提升效率、降低显存。
  ```
  pip install xformers
  ```

---

## 3. 主要参数简表

| 参数名                | 说明                   | 推荐/默认值      |
|-----------------------|------------------------|------------------|
| pretrained_model_name_or_path | 模型名或本地路径 | 详见官方文档     |
| torch_dtype           | 推理精度               | float16          |
| use_safetensors       | 优先用safetensors权重  | True             |
| num_inference_steps   | 采样步数               | 20~50            |
| guidance_scale        | 文本引导强度           | 7.5~12.0         |
| guidance_rescale      | 引导重标定             | 0.0              |
| scheduler             | 采样器类型             | default          |
| strength              | 编辑/修复保留原图比例  | 0.8              |
| is_random             | 是否随机种子           | True             |
| generator_seed        | 随机种子               | 42               |
| enable_model_cpu_offload | 权重转移到CPU      | False            |
| enable_sequential_cpu_offload | 顺序CPU offload | False            |
| enable_xformers_memory_efficient_attention | xformers高效注意力 | False |

---

## 4. 常见问题速查

- **下载慢/失败**：配置 hugging face 镜像，检查是否由模型权限，使用本地模型。
- **显存不足**：先开 `enable_model_cpu_offload`，再考虑 `enable_sequential_cpu_offload`。
- **推理慢**：关闭 `enable_sequential_cpu_offload` 或升级硬件。
- **xformers 安装失败**：检查 Python/CUDA 兼容，见 [官方文档](https://github.com/facebookresearch/xformers)。

---

如有疑问，欢迎提 issues 或加入交流群。
