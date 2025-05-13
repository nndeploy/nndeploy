
# 扩散模型

## 基于stable-diffusion1.5的文生图应用

### 下载模型

- [stable-diffusion-1.5-fp32](stable-diffusion-1.5-fp32): stable-diffusion-1.5-fp32, Model Type: onnx, output size: 512x512, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/stable_diffusion/fp32)

- [stable-diffusion-1.5-fp16](stable-diffusion-1.5-fp16): stable-diffusion-1.5-fp16, Model Type: onnx, output size: 512x512, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/stable_diffusion/fp16)

### 模型简介

- stable_diffusion/tokenizer/: 文本编码模型
- stable_diffusion/text_encoder/: 文本嵌入模型
- stable_diffusion/unet/: 扩散模型
- stable_diffusion/vae_decoder/: 图像解码模型

### 运行demo

***`注：请将PATH更换为自己对应的目录`***

#### 运行flag介绍

- --name: 模型名称
- --inference_type: 推理后端类型
- --device_type: 推理后端的执行设备类型
- --model_type: 模型类型
- --is_path: 模型是否为路径
- --model_value: 模型路径或模型文件
- --parallel_type: 并行类型
- --prompt: 输入提示词
- --output_path: 输出图片路径
- --model_inputs: 模型输入
- --model_outputs: 模型输出

#### 推理后端为onnxruntime，推理执行设备为Arm

```shell
# 进入目录
cd $nndeploy/build/

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$onnxruntime/lib:$LD_LIBRARY_PATH

# 执行
./nndeploy_demo_stable_diffusion --name txt2img --prompt "a great apple" --parallel_type kParallelTypeSequential --output_path apple.png --is_path --model_value /home/lds/stable-diffusion.onnx/models/fp32/tokenizer/tokenizer.json,/home/lds/stable-diffusion.onnx/models/fp32/text_encoder/model.onnx,/home/lds/stable-diffusion.onnx/models/fp32/unet/model.onnx,/home/lds/stable-diffusion.onnx/models/fp32/vae_decoder/model.onnx --device_type kDeviceTypeCodeCuda:0

TimeProfiler: demo
---------------------------------------------------------------------------------------------
name                               call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
---------------------------------------------------------------------------------------------
graph->init()                      1           2204.362           2204.362           0.000 
graph->dump()                      1           0.040              0.040              0.000 
graph->run()                       1           38755.879          38755.879          0.000 
txt2img run()                      1           38755.875          38755.875          0.000 
clip run()                         1           22.094             22.094             0.000 
negative_embedding_subgraph run()  1           16.281             16.281             0.000 
negative_tokenizer run()           1           0.071              0.071              0.000 
cvt_token_ids_2_tensor run()       2           0.018              0.009              0.000 
clip_infer run()                   2           21.762             10.881             0.000 
embedding_subgraph run()           1           5.639              5.639              0.000 
tokenizer run()                    1           0.058              0.058              0.000 
concat_node run()                  1           0.157              0.157              0.000 
denoise_ddim run()                 1           18709.744          18709.744          0.000 
init_latents run()                 50          0.660              0.013              0.000 
unet run()                         50          18686.123          373.722            0.000 
ddim_schedule run()                50          21.774             0.435              0.000 
vae run()                          1           20013.270          20013.270          0.000 
scale_latents run()                1           0.011              0.011              0.000 
vae_infer run()                    1           20013.252          20013.252          0.000 
save_node run()                    1           10.762             10.762             0.000 
graph->deinit()                    1           0.024              0.024              0.000 
---------------------------------------------------------------------------------------------

./nndeploy_demo_stable_diffusion --name txt2img --prompt "a great apple" --parallel_type kParallelTypeSequential --output_path apple.png --is_path --model_value /home/lds/stable-diffusion.onnx/models/fp32/tokenizer/tokenizer.json,/home/lds/stable-diffusion.onnx/models/fp32/text_encoder/model.onnx,/home/lds/stable-diffusion.onnx/models/fp32/unet/model.onnx,/home/lds/stable-diffusion.onnx/models/fp32/vae_decoder/model.onnx --device_type kDeviceTypeCodeCuda:0

TimeProfiler: demo
---------------------------------------------------------------------------------------------
name                               call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
---------------------------------------------------------------------------------------------
graph->init()                      1           2140.009           2140.009           0.000 
graph->dump()                      1           0.104              0.104              0.000 
graph->run()                       1           8890.673           8890.673           0.000 
txt2img run()                      1           8890.621           8890.621           0.000 
clip run()                         1           59.058             59.058             0.000 
negative_embedding_subgraph run()  1           56.408             56.408             0.000 
negative_tokenizer run()           1           0.085              0.085              0.000 
cvt_token_ids_2_tensor run()       2           0.022              0.011              0.000 
clip_infer run()                   2           58.734             29.367             0.000 
embedding_subgraph run()           1           2.543              2.543              0.000 
tokenizer run()                    1           0.097              0.097              0.000 
concat_node run()                  1           0.103              0.103              0.000 
denoise_ddim run()                 1           6112.537           6112.537           0.000 
init_latents run()                 1           0.333              0.333              0.000 
denoise run()                      1           6112.197           6112.197           0.000 
vae run()                          1           2700.945           2700.945           0.000 
scale_latents run()                1           0.022              0.022              0.000 
vae_infer run()                    1           2700.914           2700.914           0.000 
save_node run()                    1           18.065             18.065             0.000 
graph->deinit()                    1           17.900             17.900             0.000 
---------------------------------------------------------------------------------------------

[注] 该实现支持Sequential和Pipeline两种执行模式

### 效果示例

![sample_output](../../docs/image/demo/stable_diffusion/apple.png)