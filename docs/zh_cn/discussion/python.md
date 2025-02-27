
## python

### 纯c接口

纯c接口写在include/source中，其中需要对外导出的纯c接口，在include中导出c_xxx.h

纯c接口的写法为

nndeployBaseGetDevice()

### python接口

只包含pybind相关

### 初步完成python模块的导出，review当前实现

#### 继承 or 组合

- 该类在python侧还要重载，采用继承
  - 例如Node

- pybind那侧全部都是继承

- 组合并不会影响继承，可以继承pybind导出的类

- 组合导致继承的类还要写相关处理函数
  - 例如Node 和 Edge

- 该类基本为最终类，不会被继承
  - 例如Edge

- 该类需要被继承
  - 例如Node

#### 裸指针 or 智能指针

- 不影响性能，能改成智能指针的全部改成智能指针（要测试一下呀，去4060上跑一下）
  - 例如Param
  - 例如Inference

#### 类型强转

#### 如何解决模板问题

#### 返回值策略

#### 易用性

- 参考pytorch的接口设计

- DataType的易用性（建议用字符串构造）
- DeviceType的易用性（建议用字符串构造）
- Status的易用性（建议用字符串构造）

### 对于python端用不到的接口，不导出

### 对于目前已经导出的接口增加详细的参数列表

### 解决崩溃问题



