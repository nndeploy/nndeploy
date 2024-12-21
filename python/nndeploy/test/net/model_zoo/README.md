# test_resnet

```
cd /your_path/nndeploy/python/nndeploy/tests/net/model_zoo

python test_resnet.py --model_type onnx --model_path resnet50-v1-7.sim.onnx --device ascendcl --image_path /home/resource/data_set/example_input.jpg

python test_resnet.py --model_type onnx --model_path resnet50.onnx --device cpu --image_path /home/resource/data_set/example_input.jpg

python test_resnet.py --model_type default --model_path resnet50.json resnet50.safetensors --device cpu --image_path /home/resource/data_set/example_input.jpg   > log.txt

./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeCpu:0 --model_type kModelTypeDefault --is_path --model_value resnet50.json,resnet50.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_acl_default.jpg

```