#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;

// 演示不同的 Python 对象生命周期管理方式
class PythonObjectManager {
private:
    PyObject* stored_obj_;  // 存储的原始 PyObject 指针
    
public:
    PythonObjectManager() : stored_obj_(nullptr) {}
    
    ~PythonObjectManager() {
        if (stored_obj_) {
            Py_DECREF(stored_obj_);  // 释放引用
        }
    }
    
    // 方法1：使用 reinterpret_steal - 转移所有权
    void store_with_steal(py::object obj) {
        if (stored_obj_) {
            Py_DECREF(stored_obj_);
        }
        
        // obj.release() 返回 PyObject* 并放弃所有权
        // 我们使用 reinterpret_steal 来获取这个所有权
        PyObject* raw_ptr = obj.release().ptr();
        
        // 此时我们拥有了对象的所有权，需要手动管理
        stored_obj_ = raw_ptr;
        // 注意：不需要 Py_INCREF，因为我们"偷取"了引用
    }
    
    // 方法2：使用传统方式 - 增加引用计数
    void store_with_incref(py::object obj) {
        if (stored_obj_) {
            Py_DECREF(stored_obj_);
        }
        
        stored_obj_ = obj.ptr();
        Py_INCREF(stored_obj_);  // 增加引用计数
    }
    
    // 方法3：获取时使用 reinterpret_borrow - 不获取所有权
    py::object get_with_borrow() {
        if (!stored_obj_) {
            return py::none();
        }
        
        // 使用 reinterpret_borrow，不增加引用计数
        // 返回的对象依赖于原始对象的生命周期
        return py::reinterpret_borrow<py::object>(stored_obj_);
    }
    
    // 方法4：获取时使用 reinterpret_steal - 转移所有权
    py::object get_with_steal() {
        if (!stored_obj_) {
            return py::none();
        }
        
        // 创建对象的副本，增加引用计数
        Py_INCREF(stored_obj_);
        
        // 使用 reinterpret_steal 转移这个新引用的所有权
        return py::reinterpret_steal<py::object>(stored_obj_);
    }
    
    // 方法5：获取时使用 py::cast - 自动管理
    py::object get_with_cast() {
        if (!stored_obj_) {
            return py::none();
        }
        
        // py::cast 会自动增加引用计数
        return py::cast(stored_obj_);
    }
};

// 演示 Python C API 函数的使用
py::object create_list_with_steal() {
    // PyList_New 返回新引用
    PyObject* list = PyList_New(0);
    if (!list) {
        throw std::runtime_error("Failed to create list");
    }
    
    // 使用 reinterpret_steal 获取这个新引用的所有权
    return py::reinterpret_steal<py::object>(list);
}

py::object create_list_with_cast() {
    // PyList_New 返回新引用
    PyObject* list = PyList_New(0);
    if (!list) {
        throw std::runtime_error("Failed to create list");
    }
    
    // 使用 py::cast，但需要手动释放原始引用
    py::object result = py::cast(list);
    Py_DECREF(list);  // 释放原始引用，因为 py::cast 已经增加了引用计数
    return result;
}

// 错误示例：引用泄漏
py::object create_list_wrong() {
    PyObject* list = PyList_New(0);
    if (!list) {
        throw std::runtime_error("Failed to create list");
    }
    
    // 错误：py::cast 增加了引用计数，但我们没有释放原始引用
    // 这会导致引用泄漏
    return py::cast(list);  // 引用泄漏！
}

PYBIND11_MODULE(lifecycle_demo, m) {
    py::class_<PythonObjectManager>(m, "PythonObjectManager")
        .def(py::init<>())
        .def("store_with_steal", &PythonObjectManager::store_with_steal)
        .def("store_with_incref", &PythonObjectManager::store_with_incref)
        .def("get_with_borrow", &PythonObjectManager::get_with_borrow,
             py::return_value_policy::reference)  // 重要：使用 reference 策略
        .def("get_with_steal", &PythonObjectManager::get_with_steal)
        .def("get_with_cast", &PythonObjectManager::get_with_cast);
    
    m.def("create_list_with_steal", &create_list_with_steal);
    m.def("create_list_with_cast", &create_list_with_cast);
    m.def("create_list_wrong", &create_list_wrong);
}

/*
使用示例：

import lifecycle_demo

# 创建管理器
manager = lifecycle_demo.PythonObjectManager()

# 存储对象
my_list = [1, 2, 3]
manager.store_with_incref(my_list)

# 获取对象（不同方式）
borrowed = manager.get_with_borrow()  # 借用引用
stolen = manager.get_with_steal()     # 新的拥有引用
casted = manager.get_with_cast()      # 自动管理的引用

# 创建列表（不同方式）
list1 = lifecycle_demo.create_list_with_steal()  # 正确
list2 = lifecycle_demo.create_list_with_cast()   # 正确
list3 = lifecycle_demo.create_list_wrong()       # 引用泄漏！
*/ 