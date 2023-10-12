# TODO

./demo_nndeploy_segment --name NNDEPLOY_SAM \
                       --inference_type kInferenceTypeMnn \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeMnn \
                       --is_path \
                       --input_type kInputTypeImage \
                       --input_path C:\huggingface\nndeploy\test_data\detect\sample.jpg \
                       --output_path C:\huggingface\nndeploy\test_data\detect\sample_output.jpg


--name NNDEPLOY_SAM --inference_type kInferenceTypeMnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeMnn --is_path --input_type kInputTypeImage --input_path C:\huggingface\nndeploy\test_data\detect\sample.jpg --output_path C:\huggingface\nndeploy\test_data\detect\sample_output.jpg

--name NNDEPLOY_SAM --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\Users\59595\Downloads\embed_vitb_fp32.onnx,C:\Users\59595\Downloads\segment_vitb_fp32.onnx --input_type kInputTypeImage --input_path C:\huggingface\nndeploy\test_data\detect\sample.jpg --output_path C:\huggingface\nndeploy\test_data\detect\sample_output.jpg

--name NNDEPLOY_SAM --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\Users\59595\Downloads\embed_vitb_fp32.onnx,C:\Users\59595\Downloads\segment_vitb_fp32.onnx --input_type kInputTypeImage --input_path C:\github\mnn-segment-anything\resource\truck.jpg --output_path C:\github\mnn-segment-anything\resource\truck_result.jpg

./demo_nndeploy_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/Downloads/embed_vitb_fp32.onnx,/home/always/Downloads/segment_vitb_fp32.onnx --input_type kInputTypeImage --input_path /home/always/github/mnn-segment-anything/resource/truck.jpg --output_path /home/always/github/mnn-segment-anything/resource/truck_result.jpg


./demo_nndeploy_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/Downloads/embed_vitb_fp32.onnx,/home/always/Downloads/segment_vitb_fp32.onnx --input_type kInputTypeImage --input_path /home/always/github/mnn-segment-anything/resource/truck.jpg --output_path /home/always/github/mnn-segment-anything/resource/truck_result.jpg


./demo_nndeploy_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/Downloads/embed_vitb_fp32_sim.onnx,/home/always/Downloads/segment_vitb_fp32_sim.onnx --input_type kInputTypeImage --input_path /home/always/github/mnn-segment-anything/resource/truck.jpg --output_path /home/always/github/mnn-segment-anything/resource/truck_result.jpg