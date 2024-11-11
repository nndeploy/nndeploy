from functools import wraps

def with_net_instance(func):
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 在调用函数之前，遍历self的所有属性
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            # 检查属性是否是Conv类的实例并且是否有net_instance属性
            if isinstance(attr, Module) and hasattr(attr, 'net_instance'):
                attr.net_instance = self.model_desc
                
                
        # 对输入执行MakeInput标记
        result = func(self, *args, **kwargs)  # 调用原始函数并保存返回值
        
        # 对输出执行MakeOutPut标记
        
        # 在函数退出时执行的操作
        self.model_desc.init()  # 假设model_desc有一个init方法
        
        return result
    return wrapper


class Module():
    def __init__(self):
        self.net_instance = None  # 初始化为None


class Conv(Module):
    def __init__(self, kernel_size):
        super().__init__() 
        self.kernel_size = kernel_size
        self.conv_param=""
        

    def __call__(self, data):
        # 在调用卷积操作之前，可以访问net_instance
        self.MakeConv()
        
        # 假设这里是卷积操作的实现
        print(f"Applying convolution with kernel size {self.kernel_size}")
        return data  # 这里应该返回卷积操作的结果

    def MakeConv(self):
        # 这里可以访问Net实例
        if self.net_instance:
            print(f"Making a new convolution operation for {self.net_instance}")
        

class Relu(Module):
    
    def __init__(self):
        super().__init__() 
    
    def __call__(self, data):
        # 假设这里是ReLU激活函数的实现
        self.MakeRelu()
        print("Applying ReLU activation")
        return data
    
    def MakeRelu(self):
         if self.net_instance:
            print(f"Making a new relu operation for {self.net_instance}")

class Net():
    def __init__(self):
        self.conv1 = Conv(kernel_size=(3, 3))
        self.conv2 = Conv(kernel_size=(3, 3))
        self.relu = Relu()
        self.model_desc = nndeploy.ir.ModelDesc

    @with_net_instance
    def construct(self, data):
        # 调用卷积操作
        x = self.conv1(data)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# 使用示例
net = Net()
data = "some input data"  # 假设这是输入数据
output = net.construct(data)