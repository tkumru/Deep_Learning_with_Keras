import numpy as np
from keras.datasets import mnist

# VECTOR ----------------------------

x = np.array([1, 2, 3])
print("Vector")
print(f"Vector array: \n{x}")
print(f"Vector dimension: {x.ndim}")
print()
# MATRIX ---------------------------
print()
x = np.array([[1,2,3],[2,3,4], [3,4,5]])
print("Matrix")
print(f"Matrix array: \n{x}")
print(f"Matrix dimension: {x.ndim}")
print()
# TENSOR --------------------------------
print()
x = np.array([[[1,2,3],[2,3,4]], [[3,4,5], [4,5,6]], [[5,6,7], [6,7,8]]])
print("Tensor")
print(f"Tensor array: \n{x}")
print(f"Tensor dimension: {x.ndim}")
print()
