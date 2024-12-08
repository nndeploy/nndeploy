# TODO



./nndeploy_demo_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /data/sjx/code/nndeploy_resource/nndeploy/model_zoo/segment/sam/image_encoder_sim.onnx,/data/sjx/code/nndeploy_resource/nndeploy/model_zoo/segment/sam/sam_sim.onnx --codec_flag kCodecFlagImage --input_path /data/sjx/code/nndeploy_resource/nndeploy/test_data/detect/sample.jpg --output_path /data/sjx/code/nndeploy/build/sam_result.jpg




--name NNDEPLOY_SAM --inference_type kInferenceTypeMnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeMnn --is_path --input_type kInputTypeImage --input_path C:\huggingface\nndeploy\test_data\detect\sample.jpg --output_path C:\huggingface\nndeploy\test_data\detect\sample_output.jpg

--name NNDEPLOY_SAM --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\Users\59595\Downloads\embed_vitb_fp32.onnx,C:\Users\59595\Downloads\segment_vitb_fp32.onnx --input_type kInputTypeImage --input_path C:\huggingface\nndeploy\test_data\detect\sample.jpg --output_path C:\huggingface\nndeploy\test_data\detect\sample_output.jpg

--name NNDEPLOY_SAM --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\Users\59595\Downloads\embed_vitb_fp32.onnx,C:\Users\59595\Downloads\segment_vitb_fp32.onnx --input_type kInputTypeImage --input_path C:\github\mnn-segment-anything\resource\truck.jpg --output_path C:\github\mnn-segment-anything\resource\truck_result.jpg

./nndeploy_demo_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/Downloads/embed_vitb_fp32.onnx,/home/always/Downloads/segment_vitb_fp32.onnx --input_type kInputTypeImage --input_path /home/always/github/mnn-segment-anything/resource/truck.jpg --output_path /home/always/github/mnn-segment-anything/resource/truck_result.jpg


./nndeploy_demo_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/Downloads/embed_vitb_fp32.onnx,/home/always/Downloads/segment_vitb_fp32.onnx --input_type kInputTypeImage --input_path /home/always/github/mnn-segment-anything/resource/truck.jpg --output_path /home/always/github/mnn-segment-anything/resource/truck_result.jpg


./nndeploy_demo_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/Downloads/embed_vitb_fp32_sim.onnx,/home/always/Downloads/segment_vitb_fp32_sim.onnx --input_type kInputTypeImage --input_path /home/always/github/mnn-segment-anything/resource/truck.jpg --output_path /home/always/github/mnn-segment-anything/resource/truck_result.jpg

./nndeploy_demo_segment --name NNDEPLOY_RMBGV1.4 --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeCpu:0 --model_type kModelTypeDefault --is_path --model_value RMBGV14.json,RMBGV1.4.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path bus.jpg --output_path seg_bus_output.jpg

### 内部推理框架 - acl
./nndeploy_demo_segment --name NNDEPLOY_RMBGV1.4 --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value RMBGV1.4.json,RMBGV1.4.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_segment_acl_default.jpg

TimeProfiler: segment time profiler
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   1403.327            1403.327            0.000               
graph->run()        1                   10511.625           10511.625           0.000               
demo run()          100                 10511.385           105.114             0.000               
decode_node run()   100                 1235.786            12.358              0.000               
NNDEPLOY_RMBGV1.4 run()100                 7625.265            76.253              0.000               
preprocess run()    100                 1197.065            11.971              0.000               
infer run()         100                 5304.038            53.040              0.000               
net->run()          100                 403.422             4.034               0.000               
postprocess run()   100                 1120.696            11.207              0.000               
DrawMaskNode run()  100                 467.405             4.674               0.000               
encode_node run()   100                 1178.735            11.787              0.000               
graph->deinit()     1                   236.804             236.804             0.000               
-------------------------------------------------------------------------------------------

### 华为昇腾
./nndeploy_demo_segment --name NNDEPLOY_RMBGV1.4 --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value RMBGV1.4.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_segment_acl.jpg

atc --model=./RMBGV1.4.onnx --output=./RMBGV1.4.onnx.om --framework=5 --soc_version=Ascend910B4 --input_shape="input:1,3,1024,1024"

TimeProfiler: segment time profiler
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   1758.011            1758.011            0.000               
graph->run()        1                   6601.770            6601.770            0.000               
demo run()          100                 6601.525            66.015              0.000               
decode_node run()   100                 1204.174            12.042              0.000               
NNDEPLOY_RMBGV1.4 run()100                 3777.040            37.770              0.000               
preprocess run()    100                 1105.895            11.059              0.000               
infer run()         100                 1710.714            17.107              0.000               
postprocess run()   100                 956.797             9.568               0.000               
DrawMaskNode run()  100                 472.560             4.726               0.000               
encode_node run()   100                 1142.783            11.428              0.000               
graph->deinit()     1                   40.627              40.627              0.000               
-------------------------------------------------------------------------------------------

### ort
./nndeploy_demo_segment --name NNDEPLOY_RMBGV1.4 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeOnnx --is_path --model_value RMBGV1.4.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_segment_art.jpg

TimeProfiler: segment time profiler
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   387.103             387.103             0.000               
graph->run()        1                   392030.062          392030.062          0.000               
demo run()          100                 392029.531          3920.295            0.000               
decode_node run()   100                 1240.912            12.409              0.000               
NNDEPLOY_RMBGV1.4 run()100                 389118.781          3891.188            0.000               
preprocess run()    100                 1764.408            17.644              0.000               
infer run()         100                 386167.250          3861.673            0.000               
postprocess run()   100                 1183.069            11.831              0.000               
DrawMaskNode run()  100                 493.201             4.932               0.000               
encode_node run()   100                 1170.382            11.704              0.000               
graph->deinit()     1                   0.076               0.076               0.000               
-------------------------------------------------------------------------------------------