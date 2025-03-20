import numpy as np
from nndeploy.test.test_util import create_tensor_from_numpy, create_numpy_from_tensor

x = np.int32(123)
y = np.int32(4)
z = np.int32(79)
w  = np.int32(32)
n = np.int32(48)




x_tensor = create_tensor_from_numpy(np.array([123],dtype=np.uint8))
y_tensor = create_tensor_from_numpy(np.array([4],dtype=np.uint8))
z_tensor = create_tensor_from_numpy(np.array([79],dtype=np.int32))
w_tensor = create_tensor_from_numpy(np.array([32],dtype=np.int32))
n_tensor = create_tensor_from_numpy(np.array([48],dtype=np.int32))

x_a=create_numpy_from_tensor(x_tensor)
y_a=create_numpy_from_tensor(y_tensor)
# z_a=create_numpy_from_tensor(z_tensor)

# print(x)
# print(x_tensor)
print(x_a)
# print("xxxxxxxx")
# print(y)
# print(y_tensor)
print(y_a)
