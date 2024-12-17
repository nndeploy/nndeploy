# test_resnet

```
cd /your_path/nndeploy/python/nndeploy/tests/net/model_zoo

python test_resnet.py --model_type onnx --model_path resnet50-v1-7.sim.onnx --device cpu --image_path /data/public_dataset/imagenet/data/fhldata/ImageNet/train/n01491361/n01491361_6.JPEG

```
```
python test_resnet.py --model_type onnx --model_path /home/ascenduserdg01/github/nndeploy/build/resnet50-v1-7.sim.onnx --device cpu --image_path /home/ascenduserdg01/github/nndeploy/build/example_input.jpg

python test_resnet.py --model_type onnx --model_path /home/ascenduserdg01/github/nndeploy/build/resnet50-v1-7.sim.onnx --device ascendcl --image_path /home/ascenduserdg01/github/nndeploy/build/example_input.jpg
```