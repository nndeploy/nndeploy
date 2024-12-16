# pybind11

pybind11 是一个用于将 C++ 代码与 Python 代码进行交互的库。它允许我们在 C++ 中定义函数，然后在 Python 中调用这些函数。

## 简介
+ 单继承和多重继承
  - dag::Node可以导出为py端Node，然后重写Node
  - 新建一个torch的推理Node与 infer::Node关联起来
+ 通过Python的buffer协议，可以很轻松地获取自定义类型的内存指针。这样，我们可以很方便地在C++矩阵类型（如Eigen）和NumPy之间快速转换，而无需昂贵的拷贝操作
  - numpy与device::Tensor的转换
  - cv::Mat与device::Tensor的转换
+ 可以轻松地让C++类型支持Python pickle和unpickle操作
  - torch的pt模型为pickle格式，如何利用这个特性呢？


## 安装[链接](https://charlottelive.github.io/pybind11-Chinese-docs/03.%E5%AE%89%E8%A3%85%E8%AF%B4%E6%98%8E.html)

## 首次尝试
+ Note：在Visual Studio 2017(MSVC 14.1)上使用C++17时，pybind11需要添加标识/permissive-来让编译器强制标准一致。在Visual Studio 2019上，不做强制要求，但同样建议添加。
+ 原生支持大量数据类型，完美适用于函数参数，参数值通常直接返回或者经过py::cast处理再返回。有关完整概述，请参阅类型转换部分。
 ```
 PYBIND11_MODULE(example, m) {
    m.attr("the_answer") = 42;
    py::object world = py::cast("World");
    m.attr("what") = world;
 }
 ```

## 面对对象编程
 ```
 struct Pet {
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    std::string name;
 };

 #include <pybind11/pybind11.h>
 namespace py = pybind11;

 PYBIND11_MODULE(example, m) {
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName);
 } 
 ```
+ See also：静态成员函数需要使用class_::def_static来绑定
+ 绑定匿名函数
```
py::class_<Pet>(m, "Pet")
    .def(py::init<const std::string &>())
    .def("setName", &Pet::setName)
    .def("getName", &Pet::getName)
    .def("__repr__",
        [](const Pet &a) {
            return "<example.Pet named '" + a.name + "'>";
        });

<example.Pet named 'Molly'>
```
+ `动态属性`:要让C++类也支持动态属性，我们需要在py::class_的构造函数添加py::dynamic_attr标识
```
py::class_<Pet>(m, "Pet", py::dynamic_attr())
    .def(py::init<>())
    .def_readwrite("name", &Pet::name);

>>> p = example.Pet()
>>> p.name = "Charly"  # OK, overwrite value in C++
>>> p.age = 2  # OK, dynamically add a new attribute
>>> p.__dict__  # just like a native Python class
{'age': 2}
```
+ 需要提醒一下，支持动态属性会带来小小的运行时开销。不仅仅因为增加了额外的__dict__属性，还因为处理循环引用时需要花费更多的垃圾收集跟踪花销。但是不必担心这个问题，因为原生Python类也有同样的开销。默认情况下，pybind11导出的类比原生Python类效率更高，使能动态属性也只是让它们处于同等水平而已。

## 函数

返回值策略	描述
return_value_policy::take_ownership	引用现有对象（不创建一个新对象），并获取所有权。在引用计数为0时，Pyhton将调用析构函数和delete操作销毁对象。
return_value_policy::copy	拷贝返回值，这样Python将拥有拷贝的对象。该策略相对来说比较安全，因为两个实例的生命周期是分离的。
return_value_policy::move	使用std::move来移动返回值的内容到新实例，新实例的所有权在Python。该策略相对来说比较安全，因为两个实例的生命周期是分离的。
return_value_policy::reference	引用现有对象，但不拥有所有权。C++侧负责该对象的生命周期管理，并在对象不再被使用时负责析构它。注意：当Python侧还在使用引用的对象时，C++侧删除对象将导致未定义行为。
return_value_policy::reference_internal	返回值的生命周期与父对象的生命周期相绑定，即被调用函数或属性的this或self对象。这种策略与reference策略类似，但附加了keep_alive<0, 1>调用策略保证返回值还被Python引用时，其父对象就不会被垃圾回收掉。这是由def_property、def_readwrite创建的属性getter方法的默认返回值策略。
return_value_policy::automatic	当返回值是指针时，该策略使用return_value_policy::take_ownership。反之对左值和右值引用使用return_value_policy::copy。请参阅上面的描述，了解所有这些不同的策略的作用。这是py::class_封装类型的默认策略。
return_value_policy::automatic_reference	和上面一样，但是当返回值是指针时，使用return_value_policy::reference策略。这是在C++代码手动调用Python函数和使用pybind11/stl.h中的casters时的默认转换策略。你可能不需要显式地使用该策略。

+ keep_alive<T, N>	- 不是很理解，需要下次再详细看看

+ Call guard - 不是很理解，需要下次再详细看看

+ py::args继承自py::tuple，py::kwargs继承自py::dict

+ Non-converting参数

## 类
+ 跳板(trampoline)"的工具来重定向虚函数调用到Python中
+ 定义纯虚函数时需要使用PYBIND11_OVERRIDE_PURE宏
+ 而有默认实现的虚函数则使用PYBIND11_OVERRIDE
+ PYBIND11_OVERRIDE_PURE_NAME 和PYBIND11_OVERRIDE_NAME 宏的功能类似，主要用于C函数名和Python函数名不一致的时候。以__str__为例：
```
std::string toString() override {
  PYBIND11_OVERRIDE_NAME(
      std::string, // Return type (ret_type)
      Animal,      // Parent class (cname)
      "__str__",   // Name of method in Python (name)
      toString,    // Name of function in C++ (fn)
  );
}
```

## 异常

## 智能指针

+ py::class_<Example, std::shared_ptr<Example> /* <- holder type */> obj(m, "Example") 与 py::class_<Example> obj(m, "Example") 的异同点详细分析
  - nndeploy的指针问题 - 要不要用的问题 - 性能问题
  - share_ptr 和 unique_ptr 的性能问题 - 


## 类型转换

## python的C++接口（暂时对我们没什么用）

## 内嵌解释器（暂时对我们没什么用）

## 杂项

## 模板问题如何解决

## 参考文档
+ [pybind11 官方文档](https://pybind11.readthedocs.io/en/stable/)
+ [pybind11 中文文档](https://charlottelive.github.io/pybind11-Chinese-docs/)