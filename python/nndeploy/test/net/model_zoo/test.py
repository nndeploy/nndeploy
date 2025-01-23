import numpy as np



file1="no_opt.npy"
file2="opt.npy"

array1 = np.load(file1)
array2 = np.load(file2)

print(array1-array2)