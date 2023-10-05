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


./demo_nndeploy_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --input_type kInputTypeImage --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg


./demo_nndeploy_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --input_type kInputTypeImage --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg


./demo_nndeploy_segment --name NNDEPLOY_SAM --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --input_type kInputTypeImage --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg