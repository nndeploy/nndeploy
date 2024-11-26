# TODO



./nndeploy_demo_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /data/sjx/code/nndeploy_resource/nndeploy/model_zoo/segment/sam/image_encoder_sim.onnx,/data/sjx/code/nndeploy_resource/nndeploy/model_zoo/segment/sam/sam_sim.onnx --codec_flag kCodecFlagImage --input_path /data/sjx/code/nndeploy_resource/nndeploy/test_data/detect/sample.jpg --output_path /data/sjx/code/nndeploy/build/sam_result.jpg




--name NNDEPLOY_SAM --inference_type kInferenceTypeMnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeMnn --is_path --input_type kInputTypeImage --input_path C:\huggingface\nndeploy\test_data\detect\sample.jpg --output_path C:\huggingface\nndeploy\test_data\detect\sample_output.jpg

--name NNDEPLOY_SAM --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\Users\59595\Downloads\embed_vitb_fp32.onnx,C:\Users\59595\Downloads\segment_vitb_fp32.onnx --input_type kInputTypeImage --input_path C:\huggingface\nndeploy\test_data\detect\sample.jpg --output_path C:\huggingface\nndeploy\test_data\detect\sample_output.jpg

--name NNDEPLOY_SAM --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\Users\59595\Downloads\embed_vitb_fp32.onnx,C:\Users\59595\Downloads\segment_vitb_fp32.onnx --input_type kInputTypeImage --input_path C:\github\mnn-segment-anything\resource\truck.jpg --output_path C:\github\mnn-segment-anything\resource\truck_result.jpg

./nndeploy_demo_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/Downloads/embed_vitb_fp32.onnx,/home/always/Downloads/segment_vitb_fp32.onnx --input_type kInputTypeImage --input_path /home/always/github/mnn-segment-anything/resource/truck.jpg --output_path /home/always/github/mnn-segment-anything/resource/truck_result.jpg


./nndeploy_demo_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/Downloads/embed_vitb_fp32.onnx,/home/always/Downloads/segment_vitb_fp32.onnx --input_type kInputTypeImage --input_path /home/always/github/mnn-segment-anything/resource/truck.jpg --output_path /home/always/github/mnn-segment-anything/resource/truck_result.jpg


./nndeploy_demo_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/Downloads/embed_vitb_fp32_sim.onnx,/home/always/Downloads/segment_vitb_fp32_sim.onnx --input_type kInputTypeImage --input_path /home/always/github/mnn-segment-anything/resource/truck.jpg --output_path /home/always/github/mnn-segment-anything/resource/truck_result.jpg

./nndeploy_demo_segment --name NNDEPLOY_RMBGV1.4 --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeCpu:0 --model_type kModelTypeDefault --is_path --model_value RMBGV14.json,RMBGV1.4.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path bus.jpg --output_path seg_bus_output.jpg

./nndeploy_demo_segment --name NNDEPLOY_RMBGV1.4 --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value RMBGV1.4.json,RMBGV1.4.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path seg_example_output.jpg

atc --model=./RMBGV1.4.onnx --output=./RMBGV1.4.onnx.om --framework=5 --soc_version=Ascend910B4 --input_shape="input:1,3,1024,1024"

./nndeploy_demo_segment --name NNDEPLOY_RMBGV1.4 --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value RMBGV1.4.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path seg_example_input.jpg