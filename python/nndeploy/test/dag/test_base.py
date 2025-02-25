import nndeploy._nndeploy_internal as _C
import nndeploy.dag


# python3 nndeploy/test/dag/test_base.py


def test_type():
    
    # 测试 EdgeTypeFlag 枚举类型
    assert nndeploy.dag.EdgeTypeFlag.kBuffer == _C.dag.EdgeTypeFlag.kBuffer
    assert nndeploy.dag.EdgeTypeFlag.kCvMat == _C.dag.EdgeTypeFlag.kCvMat
    assert nndeploy.dag.EdgeTypeFlag.kTensor == _C.dag.EdgeTypeFlag.kTensor
    assert nndeploy.dag.EdgeTypeFlag.kParam == _C.dag.EdgeTypeFlag.kParam
    assert nndeploy.dag.EdgeTypeFlag.kAny == _C.dag.EdgeTypeFlag.kAny
    assert nndeploy.dag.EdgeTypeFlag.kNone == _C.dag.EdgeTypeFlag.kNone

    # 测试 name_to_edge_type_flag 字典
    assert nndeploy.dag.name_to_edge_type_flag["kBuffer"] == _C.dag.EdgeTypeFlag.kBuffer
    assert nndeploy.dag.name_to_edge_type_flag["kCvMat"] == _C.dag.EdgeTypeFlag.kCvMat
    assert nndeploy.dag.name_to_edge_type_flag["kTensor"] == _C.dag.EdgeTypeFlag.kTensor
    assert nndeploy.dag.name_to_edge_type_flag["kParam"] == _C.dag.EdgeTypeFlag.kParam
    assert nndeploy.dag.name_to_edge_type_flag["kAny"] == _C.dag.EdgeTypeFlag.kAny
    assert nndeploy.dag.name_to_edge_type_flag["kNone"] == _C.dag.EdgeTypeFlag.kNone

    # 测试 edge_type_flag_to_name 字典
    assert nndeploy.dag.edge_type_flag_to_name[_C.dag.EdgeTypeFlag.kBuffer] == "kBuffer"
    assert nndeploy.dag.edge_type_flag_to_name[_C.dag.EdgeTypeFlag.kCvMat] == "kCvMat"
    assert nndeploy.dag.edge_type_flag_to_name[_C.dag.EdgeTypeFlag.kTensor] == "kTensor"
    assert nndeploy.dag.edge_type_flag_to_name[_C.dag.EdgeTypeFlag.kParam] == "kParam"
    assert nndeploy.dag.edge_type_flag_to_name[_C.dag.EdgeTypeFlag.kAny] == "kAny"
    assert nndeploy.dag.edge_type_flag_to_name[_C.dag.EdgeTypeFlag.kNone] == "kNone"

    # 测试 from_name 方法
    assert isinstance(nndeploy.dag.EdgeTypeFlag.from_name("kBuffer"), _C.dag.EdgeTypeFlag)
    assert int(nndeploy.dag.EdgeTypeFlag.from_name("kBuffer")) == int(_C.dag.EdgeTypeFlag.kBuffer)
    assert isinstance(nndeploy.dag.EdgeTypeFlag.from_name("kCvMat"), _C.dag.EdgeTypeFlag)
    assert int(nndeploy.dag.EdgeTypeFlag.from_name("kCvMat")) == int(_C.dag.EdgeTypeFlag.kCvMat)
    assert isinstance(nndeploy.dag.EdgeTypeFlag.from_name("kTensor"), _C.dag.EdgeTypeFlag)
    assert int(nndeploy.dag.EdgeTypeFlag.from_name("kTensor")) == int(_C.dag.EdgeTypeFlag.kTensor)
    assert isinstance(nndeploy.dag.EdgeTypeFlag.from_name("kParam"), _C.dag.EdgeTypeFlag)
    assert int(nndeploy.dag.EdgeTypeFlag.from_name("kParam")) == int(_C.dag.EdgeTypeFlag.kParam)
    assert isinstance(nndeploy.dag.EdgeTypeFlag.from_name("kAny"), _C.dag.EdgeTypeFlag)
    assert int(nndeploy.dag.EdgeTypeFlag.from_name("kAny")) == int(_C.dag.EdgeTypeFlag.kAny)
    assert isinstance(nndeploy.dag.EdgeTypeFlag.from_name("kNone"), _C.dag.EdgeTypeFlag)
    assert int(nndeploy.dag.EdgeTypeFlag.from_name("kNone")) == int(_C.dag.EdgeTypeFlag.kNone)

    # 测试无效的类型名称
    try:
        nndeploy.dag.EdgeTypeFlag.from_name("invalid_type")
        assert False
    except ValueError:
        assert True


def test_edge_type_info():
    edge_type_info = nndeploy.dag.EdgeTypeInfo()
    edge_type_info.set_buffer_type()
    print(edge_type_info)
    print(edge_type_info.type_name)
    

if __name__ == "__main__":
    test_type()
    test_edge_type_info()