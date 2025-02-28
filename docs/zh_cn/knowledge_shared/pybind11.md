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
- return_value_policy::take_ownership	引用现有对象（不创建一个新对象），并获取所有权。在引用计数为0时，Pyhton将调用析构函数和delete操作销毁对象。
- return_value_policy::copy	拷贝返回值，这样Python将拥有拷贝的对象。该策略相对来说比较安全，因为两个实例的生命周期是分离的。
  - return_value_policy::copy要求对象支持拷贝语义（有适当的复制构造函数或复制赋值运算符）。如果对象的复制不正确实现，可能会导致资源管理问题。**所以通常是作用与对象，而非指针**
- return_value_policy::move	使用std::move来移动返回值的内容到新实例，新实例的所有权在Python。该策略相对来说比较安全，因为两个实例的生命周期是分离的。
  - return_value_policy::move要求对象支持移动语义（有适当的移动构造函数或移动赋值运算符）。如果对象的移动不正确实现，可能会导致资源管理问题。**所以通常是作用与对象，而非指针**
- return_value_policy::reference	引用现有对象，但不拥有所有权。C++侧负责该对象的生命周期管理，并在对象不再被使用时负责析构它。注意：当Python侧还在使用引用的对象时，C++侧删除对象将导致未定义行为。
- return_value_policy::reference_internal	返回值的生命周期与父对象的生命周期相绑定，即被调用函数或属性的this或self对象。这种策略与reference策略类似，但附加了keep_alive<0, 1>调用策略保证返回值还被Python引用时，其父对象就不会被垃圾回收掉。这是由def_property、def_readwrite创建的属性getter方法的默认返回值策略。
- return_value_policy::automatic	当返回值是指针时，该策略使用return_value_policy::take_ownership。反之对左值和右值引用使用return_value_policy::copy。请参阅上面的描述，了解所有这些不同的策略的作用。这是py::class_封装类型的默认策略。
- return_value_policy::automatic_reference	和上面一样，但是当返回值是指针时，使用return_value_policy::reference策略。这是在C++代码手动调用Python函数和使用pybind11/stl.h中的casters时的默认转换策略。你可能不需要显式地使用该策略。

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

# 函数返回的是智能指针还是裸指针。

首先，我们看一下 `create_decode_node` 函数的定义：

```cpp
// plugin/source/nndeploy/codec/codec.cc
DecodeNode *createDecodeNode(base::CodecType type, base::CodecFlag flag,
                             const std::string &name, dag::Edge *output) {
  DecodeNode *temp = nullptr;
  auto &map = getGlobaCreatelDecodeNodeFuncMap();
  if (map.count(type) > 0) {
    temp = map[type](flag, name, output);
  }
  return temp;
}
```

这个函数的返回类型是 `DecodeNode *`，这是一个裸指针（raw pointer），而不是智能指针。

我们再看一下 Python 绑定中此函数的注册方式：

```cpp
// python/src/codec/codec.cc
m.def("create_decode_node", &createDecodeNode, py::arg("type"),
        py::arg("flag"), py::arg("name"), py::arg("output"), py::return_value_policy::take_ownership);
```

这里使用了 `py::return_value_policy::take_ownership` 作为返回值策略。这个策略意味着 Python 端会获取返回的裸指针的所有权，当 Python 对象被销毁时，该指针也会被删除。

所以，`_C.codec.create_decode_node` 返回的是一个裸指针，但 Python 接口会接管这个指针的生命周期管理。

另一个重要的注意点是，`Node` 类的绑定使用了 `std::shared_ptr<Node>` 作为持有者类型：

```cpp
py::class_<Node, PyNode, std::shared_ptr<Node>>(m, "Node", py::dynamic_attr())
```

但是，这不意味着 `create_decode_node` 函数返回智能指针。它仍然返回裸指针，只是在 Python 绑定中，pybind11 会将这个裸指针包装成一个带有 `std::shared_ptr` 持有者类型的 Python 对象。

总结：
1. `createDecodeNode` 函数本身返回的是裸指针 `DecodeNode*`
2. 由于使用了 `py::return_value_policy::take_ownership`，Python 会接管这个指针的生命周期
3. 绑定 `Node` 类时使用了 `std::shared_ptr<Node>` 作为持有者类型，所以在 Python 端，这个裸指针会被包装成一个使用 `std::shared_ptr` 管理的对象

## Python函数返回值分析：裸指针还是智能指针

分析 `create_decode_node` 函数返回的是什么类型的指针：

### 函数定义和Python绑定

```cpp
// C++实现
DecodeNode *createDecodeNode(base::CodecType type, base::CodecFlag flag,
                             const std::string &name, dag::Edge *output) {
  DecodeNode *temp = nullptr;
  auto &map = getGlobaCreatelDecodeNodeFuncMap();
  if (map.count(type) > 0) {
    temp = map[type](flag, name, output);
  }
  return temp;
}

// Python绑定
m.def("create_decode_node", &createDecodeNode, py::arg("type"),
      py::arg("flag"), py::arg("name"), py::arg("output"), 
      py::return_value_policy::take_ownership);
```

### 分析结果

函数 `self.decode_node = _C.codec.create_decode_node(...)` 返回的是**裸指针（raw pointer）**，而非智能指针，原因如下：

1. **C++函数返回类型**：`DecodeNode *` 是裸指针类型
2. **返回值策略**：使用了 `py::return_value_policy::take_ownership`，这表示Python会接管裸指针的所有权
3. **内存管理方式**：当Python对象被析构时，会自动删除这个指针指向的对象

### 与Node类绑定方式的差异

虽然 `Node` 类的绑定使用了共享指针作为持有者类型：

```cpp
py::class_<Node, PyNode, std::shared_ptr<Node>>(m, "Node", py::dynamic_attr())
```

但这并不改变 `create_decode_node` 函数本身返回裸指针的事实。区别在于：

- `Node` 类使用 `std::shared_ptr` 作为持有者类型时，Python和C++会共享对象所有权
- 而 `create_decode_node` 返回的裸指针通过 `take_ownership` 策略，所有权转移给了Python

### 可能存在的问题

在当前实现中，`create_decode_node` 返回裸指针并使用 `take_ownership` 可能与 `Node` 类使用 `std::shared_ptr` 持有者的设计不完全一致，需要注意以下潜在问题：

1. 如果 `DecodeNode` 对象在C++端有共享所有权的需求
2. 对象的生命周期管理不一致导致的内存问题

建议考虑修改 `createDecodeNode` 返回 `std::shared_ptr<DecodeNode>` 以保持一致性，或确保当前设计在您的使用场景中是安全的。


## 好用的功能

- 通过Python的buffer协议，可以很轻松地获取自定义类型的内存指针。这样，我们可以很方便地在C++矩阵类型（如Eigen）和NumPy之间快速转换，而无需昂贵的拷贝操作。
- 只需几行代码就可以支持Python基于切片的访问和赋值操作。

### 虚基类

pet_store函数返回了一个Dog实例，但由于基类并非多态类型，Python只识别到了Pet。在C++中，一个类至少有一个虚函数才会被视为多态类型。pybind11会自动识别这种多态机制。


## 定义纯虚函数时需要使用PYBIND11_OVERRIDE_PURE宏，而有默认实现的虚函数则使用PYBIND11_OVERRIDE。PYBIND11_OVERRIDE_PURE_NAME 和PYBIND11_OVERRIDE_NAME 宏的功能类似，主要用于C函数名和Python函数名不一致的时候。以__str__为例：

## note

如果你在派生的Python类中自定义了一个构造函数，你必须保证显示调用C++构造函数(通过__init__)，不管它是否为默认构造函数。否则，实例属于C++那部分的内存就未初始化，可能导致未定义行为。在pybind11 2.6版本中，这种错误将会抛出TypeError异常。

class Dachshund(Dog):
    def __init__(self, name):
        Dog.__init__(self)  # Without this, a TypeError is raised.
        self.name = name

    def bark(self):
        return "yap!"
注意必须显式地调用__init__，而不应该使用supper()。在一些简单的线性继承中，supper()或许可以正常工作；一旦你混合Python和C++类使用多重继承，由于Python MRO和C++的机制，一切都将崩溃。


## 虚函数与继承
综合考虑虚函数与继承时，你需要为每个你允许在Python派生类中重载的方法提供重载方式。下面我们扩展Animal和Dog来举例：

class Animal {
public:
    virtual std::string go(int n_times) = 0;
    virtual std::string name() { return "unknown"; }
};
class Dog : public Animal {
public:
    std::string go(int n_times) override {
        std::string result;
        for (int i=0; i<n_times; ++i)
            result += bark() + " ";
        return result;
    }
    virtual std::string bark() { return "woof!"; }
};
上节涉及到的Animal辅助类仍是必须的，为了让Python代码能够继承Dog类，我们也需要为Dog类增加一个跳板类，来实现bark()和继承自Animal的go()、name()等重载方法（即便Dog类并不直接重载name方法）。

class PyAnimal : public Animal {
public:
    using Animal::Animal; // Inherit constructors
    std::string go(int n_times) override { PYBIND11_OVERRIDE_PURE(std::string, Animal, go, n_times); }
    std::string name() override { PYBIND11_OVERRIDE(std::string, Animal, name, ); }
};
class PyDog : public Dog {
public:
    using Dog::Dog; // Inherit constructors
    std::string go(int n_times) override { PYBIND11_OVERRIDE(std::string, Dog, go, n_times); }
    std::string name() override { PYBIND11_OVERRIDE(std::string, Dog, name, ); }
    std::string bark() override { PYBIND11_OVERRIDE(std::string, Dog, bark, ); }
};
注意到name()和bark()尾部的逗号，这用来说明辅助类的函数不带任何参数。当函数至少有一个参数时，应该省略尾部的逗号。

注册一个继承已经在pybind11中注册的带虚函数的类，同样需要为其添加辅助类，即便它没有定义或重载任何虚函数：

class Husky : public Dog {};
class PyHusky : public Husky {
public:
    using Husky::Husky; // Inherit constructors
    std::string go(int n_times) override { PYBIND11_OVERRIDE_PURE(std::string, Husky, go, n_times); }
    std::string name() override { PYBIND11_OVERRIDE(std::string, Husky, name, ); }
    std::string bark() override { PYBIND11_OVERRIDE(std::string, Husky, bark, ); }
};
我们可以使用模板辅助类将简化这类重复的绑定工作，这对有多个虚函数的基类尤其有用：

template <class AnimalBase = Animal> class PyAnimal : public AnimalBase {
public:
    using AnimalBase::AnimalBase; // Inherit constructors
    std::string go(int n_times) override { PYBIND11_OVERRIDE_PURE(std::string, AnimalBase, go, n_times); }
    std::string name() override { PYBIND11_OVERRIDE(std::string, AnimalBase, name, ); }
};
template <class DogBase = Dog> class PyDog : public PyAnimal<DogBase> {
public:
    using PyAnimal<DogBase>::PyAnimal; // Inherit constructors
    // Override PyAnimal's pure virtual go() with a non-pure one:
    std::string go(int n_times) override { PYBIND11_OVERRIDE(std::string, DogBase, go, n_times); }
    std::string bark() override { PYBIND11_OVERRIDE(std::string, DogBase, bark, ); }
};
这样，我们只需要一个辅助方法来定义虚函数和纯虚函数的重载了。只是这样编译器就需要生成许多额外的方法和类。

下面我们在pybind11中注册这些类：

py::class_<Animal, PyAnimal<>> animal(m, "Animal");
py::class_<Dog, Animal, PyDog<>> dog(m, "Dog");
py::class_<Husky, Dog, PyDog<Husky>> husky(m, "Husky");
// ... add animal, dog, husky definitions
注意，Husky不需要一个专门的辅助类，因为它没定义任何新的虚函数和纯虚函数的重载。

Python中的使用示例：

class ShihTzu(Dog):
    def bark(self):
        return "yip!"


## std::shared_ptr == 对于智能指针管理的类型，永远不要在函数如参数或返回值中使用原始指针
class_可以传递一个表示持有者类型的模板类型，它用于管理对象的引用。在不指定的情况下，默认为std::unique_ptr<Type>类型，这意味着当Python的引用计数为0时，将析构对象。该模板类型可以指定为其他的智能指针或引用计数包装类，像下面我们就使用了std::shared_ptr：

py::class_<Example, std::shared_ptr<Example> /* <- holder type */> obj(m, "Example");
注意，每个类仅能与一个持有者类型关联。

使用持有者类型的一个潜在的障碍就是，你需要始终如一的使用它们。猜猜下面的绑定代码有什么问题？

class Child { };

class Parent {
public:
   Parent() : child(std::make_shared<Child>()) { }
   Child *get_child() { return child.get(); }  /* Hint: ** DON'T DO THIS ** */
private:
    std::shared_ptr<Child> child;
};

PYBIND11_MODULE(example, m) {
    py::class_<Child, std::shared_ptr<Child>>(m, "Child");

    py::class_<Parent, std::shared_ptr<Parent>>(m, "Parent")
       .def(py::init<>())
       .def("get_child", &Parent::get_child);
}
下面的Python代码将导致未定义行为（类似段错误）。

from example import Parent
print(Parent().get_child())
问题在于Parent::get_child()返回类Child实例的指针，但事实上这个经由std::shared_ptr<...>管理的实例，在传递原始指针时就丢失了。这个例子中，pybind11将创建第二个独立的std::shared_ptr<...>声明指针的所有权。最后，对象将被free两次，因为两个shared指针没法知道彼此的存在。

有两种方法解决这个问题：

对于智能指针管理的类型，永远不要在函数如参数或返回值中使用原始指针。换句话说，在任何需要使用该类型指针的地方，使用它们指定的持有者类型代替。这个例子中get_child()可以这样修改：

std::shared_ptr<Child> get_child() { return child; }
定义Child时指定std::enable_shared_from_this<T>作为基类。这将在Child的基础上增加一点信息，让pybind11认识到这里已经存在一个std::shared_ptr<...>，并与之交互。修改示例如下：


class Child : public std::enable_shared_from_this<Child> { };

## 隐式转换
py::implicitly_convertible<A, B>();

## 参考文档
+ [pybind11 官方文档](https://pybind11.readthedocs.io/en/stable/)
+ [pybind11 中文文档](https://charlottelive.github.io/pybind11-Chinese-docs/)
