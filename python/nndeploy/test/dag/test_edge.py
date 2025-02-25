import nndeploy._nndeploy_internal as _C
import nndeploy.dag


# python3 nndeploy/test/dag/test_edge.py


def test_edge():
    edge = nndeploy.dag.Edge("test_edge")
    assert edge is not None
    print(edge.get_name())
    print(edge.get_parallel_type() == nndeploy.base.ParallelType.Sequential)
    print(edge.get_buffer(None) is None)
    # print(edge.get_cv_mat(None) is None)
    print(edge.get_tensor(None) is None)
    print(edge.get_param(None) is None)
    print(edge.get_graph_output_buffer() is None)
    # print(edge.get_graph_output_cv_mat() is None)
    print(edge.get_graph_output_tensor() is None)
    print(edge.get_graph_output_param() is None)
    
    # edge.set(1)
    # print(edge.get() == 1)
    # edge.set(1.0)
    # print(edge.get() == 1.0)
    # edge.set("test")
    # print(edge.get() == "test")
    # edge.set(nndeploy.device.Buffer(nndeploy.device.Device("cpu"), nndeploy.device.BufferDesc(1024)))
    # print(edge.get())
    tensor = nndeploy.device.Tensor()
    edge.set(tensor)
    # print(edge.get_tensor(None))
    # edge.set(nndeploy.base.Param())
    # print(edge.get())


if __name__ == "__main__":
    test_edge()
