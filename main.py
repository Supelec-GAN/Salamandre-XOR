import numpy as np
from xor.neuronLayer import *
from xor.network import *


def f(x):
    if x >= 0:
        return 1
    return -1


fn = np.vectorize(f)


nl = NeuronLayer(2, 1, fn)
nl.weights = np.array([[1], [0]])
print(nl.compute(np.array([5, 5])))
print(nl.activation_levels)


net = Network(np.array([2, 2, 1]), np.array([fn, fn]))
print(nl.compute(np.array([-5, 5])))