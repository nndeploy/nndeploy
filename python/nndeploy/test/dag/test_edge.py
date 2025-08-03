import nndeploy._nndeploy_internal as _C
import nndeploy.dag


# python3 nndeploy/test/dag/test_edge.py


def test_edge():
    edge = nndeploy.dag.Edge("test_edge")
    assert edge is not None
    print(edge.get_name())
    print(edge.get_buffer(None) is None)
    # print(edge.get_cv_mat(None) is None)
    print(edge.get_tensor(None) is None)
    print(edge.get_param(None) is None)
    print(edge.get_graph_output_buffer() is None)
    # print(edge.get_graph_output_cv_mat() is None)
    print(edge.get_graph_output_tensor() is None)
    print(edge.get_graph_output_param() is None)
    
    edge.set(1)
    print(edge.get() == 1)
    print(edge.get_type_name())
    tensor = nndeploy.device.Tensor()
    edge.set(tensor)
    print(edge.get_type_name())
    edge.create_buffer(nndeploy.device.Device("cpu")._device, nndeploy.device.BufferDesc(1024))
    print(edge.get_type_name())
    import torch
    edge.set(torch.randn(1024))
    print(edge.get_type_name())
    print(edge.get_graph_output())
    
    import cv2 as cv
    mat = cv.imread("/home/always/github/public/nndeploy/build/draw_label_node.jpg")
    edge.set(mat)
    print(edge.get_type_name())
    print(edge.get_graph_output_numpy())


if __name__ == "__main__":
    test_edge()
