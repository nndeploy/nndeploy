C++ API 参考
===============

本页面展示 nndeploy C++ API 文档。

.. note::
   此文档从源码自动生成。
   如果下面的内容为空，请检查 Breathe 配置。

快速测试
--------

.. breathe:class:: nndeploy::device::Device
   :project: nndeploy_device
   :outline:

核心类详细文档
--------------

设备信息结构
~~~~~~~~~~~~

.. breathe:struct:: nndeploy::device::DeviceInfo
   :project: nndeploy_device
   :members:

设备架构管理类
~~~~~~~~~~~~~~~~

.. breathe:class:: nndeploy::device::Architecture
   :project: nndeploy_device
   :members:
   :protected-members:

设备基类
~~~~~~~~

.. breathe:class:: nndeploy::device::Device
   :project: nndeploy_device
   :members:
   :protected-members:

流管理类
~~~~~~~~

.. breathe:class:: nndeploy::device::Stream
   :project: nndeploy_device
   :members:
   :protected-members:

事件管理类
~~~~~~~~~~

.. breathe:class:: nndeploy::device::Event
   :project: nndeploy_device
   :members:
   :protected-members:

模板类
------

.. breathe:class:: nndeploy::device::TypeArchitectureRegister
   :project: nndeploy_device
   :members:

全局函数
--------

.. breathe:function:: nndeploy::device::getDevice
   :project: nndeploy_device

.. breathe:function:: nndeploy::device::createStream(base::DeviceType)
   :project: nndeploy_device

.. breathe:function:: nndeploy::device::createEvent
   :project: nndeploy_device

完整索引
--------

.. breathe:doxygen-index::
   :project: nndeploy_device
   :outline: