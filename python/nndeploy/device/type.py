
import nndeploy._nndeploy_internal as _C

import nndeploy.base


class BufferDesc(_C.device.BufferDesc):
    def __init__(self, *args, **kwargs):
        """
        Constructs a BufferDesc object.

        The constructor can be called in the following ways:
        1. BufferDesc(): Constructs an empty BufferDesc.
        2. BufferDesc(size): Constructs a BufferDesc with size.
        3. BufferDesc(size_ptr, size): Constructs a BufferDesc from a size_t array of size.
        4. BufferDesc(size_vector): Constructs a BufferDesc from a base::SizeVector.
        5. BufferDesc(size, int_vector): Constructs a BufferDesc from a size and a base::IntVector.
        6. BufferDesc(size_vector, int_vector): Constructs a BufferDesc from a base::SizeVector and a base::IntVector.
        7. BufferDesc(size_ptr, size, int_vector): Constructs a BufferDesc from a size_t array of size and a base::IntVector.
        """
        super().__init__(*args, **kwargs)
    
    def __eq__(self, other):
        return super().__eq__(other)
    
    def __ne__(self, other):
        return super().__ne__(other)
    
    def __ge__(self, other):
        return super().__ge__(other)
    
    def get_size(self):
        return super().get_size()
    
    def get_size_vector(self):
        return super().get_size_vector()
    
    def get_real_size(self):
        return super().get_real_size()
    
    def get_real_size_vector(self):
        return super().get_real_size_vector()
    
    def get_config(self):
        return super().get_config()
    
    def is_same_config(self, other):
        return super().is_same_config(other)
    
    def is_same_dim(self, other):
        return super().is_same_dim(other)
    
    def is_1d(self):
        return super().is_1d()
    
    def print(self, stream):
        super().print(stream)
    
    def just_modify(self, *args):
        return super().just_modify(*args)
    
    def clear(self):
        super().clear()
    
    def __str__(self):
        return super().__str__()

class TensorDesc(_C.device.TensorDesc):
    def __init__(self, *args, **kwargs):
        """
        Constructs a TensorDesc object.

        The constructor can be called in the following ways:
        1. TensorDesc(): Constructs an empty TensorDesc object.
        2. TensorDesc(data_type, format, shape): Constructs a TensorDesc object from data type, data format and shape.
        3. TensorDesc(data_type, format, shape, stride): Constructs a TensorDesc object from data type, data format, shape and stride.
        4. TensorDesc(desc): Constructs a new TensorDesc object from another TensorDesc object.
        """
        super().__init__(*args, **kwargs)

    def __eq__(self, other):
        return super().__eq__(other)

    def __ne__(self, other):  
        return super().__ne__(other)

    def print(self, stream):
        super().print(stream)

    @property
    def data_type(self):
        return self.data_type_

    @data_type.setter
    def data_type(self, value):
        self.data_type_ = value

    @property
    def data_format(self):
        return self.data_format_

    @data_format.setter 
    def data_format(self, value):
        self.data_format_ = value

    @property
    def shape(self):
        return self.shape_

    @shape.setter
    def shape(self, value):  
        self.shape_ = value

    @property
    def stride(self):
        return self.stride_

    @stride.setter
    def stride(self, value):
        self.stride_ = value

    def __str__(self):
        return super().__str__()
  
    

    
    
