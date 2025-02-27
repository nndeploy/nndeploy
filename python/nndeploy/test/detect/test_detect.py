
import nndeploy._nndeploy_internal as _C

# from nndeploy._nndeploy_internal import Node, NodeDesc, Graph

import nndeploy.base
import nndeploy.device
import nndeploy.dag
import nndeploy.detect
import torch


# python3 nndeploy/test/detect/test_detect.py
    
        
    
def test_detect():
    print("test_detect")
    # 创建detect_graph
    outputs = _C.dag.Edge("outputs")
    detect_graph = nndeploy.detect.DetectGraph("detect_graph", outputs)
    
    detect_graph.set_time_profile_flag(True)
    
    detect_graph.init()
    detect_graph.dump()
    detect_graph.yolo_graph.dump()
    
    # 设置输入路径
    detect_graph.set_input_path("/home/ascenduserdg01/github/nndeploy/docs/image/demo/detect/sample.jpg")
    # # 设置输出路径
    detect_graph.set_output_path("/home/ascenduserdg01/github/nndeploy/build/aaa.jpg")
    _C.base.time_point_start("py_run")
    detect_graph.run() 
    _C.base.time_point_end("py_run")
    print("run end")  
    
    result = outputs.get_graph_output_param()
    print(result.bboxs_[0].index_)
    print(result.bboxs_[0].label_id_)
    print(result.bboxs_[0].score_)
    print(result.bboxs_[0].bbox_)
    print(result.bboxs_[0].mask_)
    detect_graph.deinit()
    
    _C.base.time_profiler_print("py_run")
    
    
if __name__ == "__main__":
    test_detect()
    
    
        
        
        
        
