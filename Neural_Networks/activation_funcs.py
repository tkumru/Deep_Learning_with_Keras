from re import A
import numpy as np
import matplotlib.pyplot as plt
import math

# Definition of Activation Functions

# SIGMOID -------------------------

def sigmoid(x):
    a = list()
    for i in x:
        a.append(1 / (1 + math.exp(-i)))

    return a

# HYPERBOLIC TANGENT -------------

def tanh(x, derivative=False):
    if derivative==True: return (1 - (x**2))
    
    return np.tanh(x)

# RELU ---------------------------

def relu(x):
    b = list()
    for i in x:
        b.append(0) if i < 0 else b.append(i)

    return b

# LEAKY RELU --------------------

def leru(x):
    b = list()
    for i in x:
        b.append(i / 10) if i < 0 else b.append(i)

    return b

# SWISH -------------------------

swish = lambda x: sigmoid(x) * x

# Defining variable for plotting

x = np.arange(-3., 3., 0.1)

_sig = sigmoid(x)
_tanh = tanh(x)
_relu = relu(x)
_lrelu = leru(x)
_swish = swish(x)

# Plotting variable

line1, = plt.plot(x, _sig, label='Sigmoid')
line2, = plt.plot(x, _tanh, label='Tanh')
line3, = plt.plot(x, _relu, label='ReLu')
line4, = plt.plot(x, _lrelu, label='Leaky ReLu')
line5, = plt.plot(x, _swish, label='Swish')

plt.legend(handles=[line1, line2, line3, line4, line5])
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()
