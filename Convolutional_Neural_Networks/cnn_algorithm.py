from turtle import shape
from cv2 import multiply
import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

# Padding pixels with zero

def zero_pad(x, pad):
    x_pad = np.pad(x, ((0,0), (pad, pad), (pad, pad), (0,0)), 'constant', constant_values=0)
    
    return x_pad

np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)

print(f"x.shape: {x.shape}")
print(f"x_pad.shape: {x_pad.shape}")
print()
print(f"x[1, 1]: \n{x[1, 1]}")
print()
print(f"x_pad[1, 1]: \n{x_pad[1, 1]}")
print()

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0, :, :, 0])
plt.show()

# Get the convolution single step

"""
a_slice_prev = It is input matrix. It has got number channels, width and length.
W = Weight matrix
b = bias variable

Number of weigh channels and number of a_slice_prev chanelles must be equal
"""

def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W)  # input and weight multiply
    Z = np.sum(s)
    Z = float(b) + Z

    return Z

np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)
Z = conv_single_step(a_slice_prev, W, b)
print("Convolution Single Step")
print(f"Z: {Z}")
print()

# Convolution Forward Supervision

def conv_forward(a_prev, W, b, hparameters):
    """
    H_prev = length of matrix
    W_prev = width of matrix
    C_prev = number of channels
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = a_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int(((n_H_prev - f + 2 * pad)/ stride) + 1)  # output dimension
    n_W = int(((n_W_prev - f + 2 * pad)/ stride) + 1)

    Z = np.zeros([m, n_H, n_W, n_C])
    A_prev_pad = zero_pad(a_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vertical_start = h * stride
                    vertical_end = vertical_start + f

                    horizantal_start = w * stride
                    horizantal_end = horizantal_start + f
                    
                    a_slice_prev = a_prev_pad[vertical_start:vertical_end, horizantal_start:horizantal_end]

                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[Ellipsis, c], b[Ellipsis, c])

    assert(Z.shape == (m, n_H, n_W, n_C))

    cache = (a_prev, W, b, hparameters)

    return Z, cache

np.random.seed(1)
a_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad": 2, "stride": 2}

Z, cache_conv = conv_forward(a_prev, W, b, hparameters)

print("Convolution Forward Supervision")
print(f"Z means: {np.mean(Z)}")
print(f"Z[3, 2, 1]: {Z[3, 2, 1]}")
print(f"cache[0][1][2][3]: {cache_conv[0][1][2][3]}")
print()

# Pooling Forward Supervision

def pool_forward(A_prev, hparameters, mode='max'):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters['f']
    stride = hparameters['stride']

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vertical_start = h * stride
                    vertical_end = vertical_start + f

                    horizantal_start = w * stride
                    horizantal_end = horizantal_start + f

                    a_slice_prev = A_prev[i , vertical_start:vertical_end,
                                         horizantal_start:horizantal_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_slice_prev)  
                    else: 
                        A[i, h, w, c] = np.mean(a_slice_prev)

    cache = (A_prev, hparameters)
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache

np.random.seed(1)
a_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride": 2, 'f': 3}
A, cache = pool_forward(a_prev, hparameters)

print("Pooling Forward Supervision")
print(f"Mode = max --> A: \n{A}")
print("-------------------------------------------" * 3)

A, cache = pool_forward(a_prev, hparameters, mode="average")
print(f"Mode = average --> A: \n{A}")
print()

# Convolution Backward Supervision

def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vertical_start = h
                    vertical_end = vertical_start + f

                    horizantal_start = w
                    horizantal_end = horizantal_start + f

                    a_slice = a_prev_pad[vertical_start:vertical_end, horizantal_start:horizantal_end, :]
                    da_prev_pad[vertical_start:vertical_end, horizantal_start:horizantal_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db

np.random.seed(1)
dA, dW, db = conv_backward(Z, cache_conv)

print("Convolution Backward Supervision")
print(f"dA means: {np.mean(dA)}")
print(f"dW means: {np.mean(dW)}")
print(f"db means: {np.mean(db)}")
print()

# Create Filter with maximum value

def create_mask_from_window(x):
    mask = x == np.max(x)

    return mask

np.random.seed(1)
x = np.random.randn(2 ,3)
mask = create_mask_from_window(x)

print("Create Filter with maximum value")
print(f"x: \n{x}")
print(f"mask: \n{mask}")
print()

# Create Filter with mean

def distribute_value(dZ, shape):
    (n_H, n_W) = shape
    average = dZ / (n_H * n_W)
    a = np.ones(shape) * average

    return a

a = distribute_value(dZ=2, shape=(2, 2))

print("Create Filter with mean value")
print(f"mean mask: \n{a}")
print()

def pool_backward(dA, cache, mode="max"):
    (A_prev, hparameters) = cache

    stride = hparameters['stride']
    f = hparameters['f']

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vertical_start = h
                    vertical_end = vertical_start + f

                    horizantal_start = w
                    horizantal_end = horizantal_start + f

                    if mode == "max":
                        a_prev_slice = a_prev[vertical_start:vertical_end, horizantal_start:horizantal_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vertical_start:vertical_end, horizantal_start:horizantal_end, c] += np.multiply(mask, dA[i, h, w, c])
                    else:
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vertical_start:vertical_end, horizantal_start:horizantal_end, c] += distribute_value(da, shape)

    assert(dA_prev.shape == A_prev.shape)

    return dA_prev

np.random.seed(1)

A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride": 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
dA = np.random.randn(5, 4, 2, 2)
dA_prev = pool_backward(dA, cache, mode='max')

print("**********************************")
print("For mode = max;")
print(f"dA mean: {np.mean(dA)}")
print(f"dA_prev[1, 1]: {dA_prev[1, 1]}")
print()

dA_prev = pool_backward(dA, cache, mode='average')
print("For mode = average;")
print(f"dA mean: {np.mean(dA)}")
print(f"dA_prev[1, 1]: {dA_prev[1, 1]}")
