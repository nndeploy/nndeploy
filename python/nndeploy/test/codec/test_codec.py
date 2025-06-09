"""
测试Codec类
"""

import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.op
import nndeploy.inference
import nndeploy.dag
import nndeploy.infer
import torch
import nndeploy.codec as codec

# python3 nndeploy/test/codec/test_codec.py
# 
# export LD_LIBRARY_PATH=/path/to/nndeploy/build:$LD_LIBRARY_PATH

def test_codec():
    # 创建输出边
    output = nndeploy.dag.Edge("output")

    # 创建解码节点
    decode_node = codec.create_decode_node(nndeploy.base.CodecType.OpenCV, nndeploy.base.CodecFlag.Image, "decode", output)
    print(decode_node)
    
    # 测试设置和获取参数
    decode_node.set_codec_flag(nndeploy.base.CodecFlag.Image)
    print(decode_node.get_codec_flag())
    
    # 设置输入路径
    decode_node.set_path("/home/always/github/public/nndeploy/build/draw_label_node.jpg")
    
    decode_node.init()
    
    # 测试运行
    status = decode_node.run()
    print(status)
    
    # 测试获取视频/图像信息
    width = decode_node.get_width()
    print(f"width: {width}")
    height = decode_node.get_height()
    print(f"height: {height}")
    fps = decode_node.get_fps()
    print(f"fps: {fps}")
    size = decode_node.get_size()
    print(f"size: {size}")
    
    print(type(decode_node))
    print(type(decode_node).__base__)
    print(type(decode_node).__base__.__base__)
    # print(type(decode_node).__base__.__base__.__base__)
    # print(type(decode_node).__base__.__base__.__base__.__base__)
    # print(type(decode_node).__base__.__base__.__base__.__base__.__base__)
    # print(type(decode_node).__base__.__base__.__base__.__base__.__base__.__base__)

if __name__ == "__main__":
    test_codec()
