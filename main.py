import numpy as np
from xor.neuronLayer import NeuronLayer
from xor.network import Network
from function.sigmoid import Sigmoid


def f(x):
    if x >= 0:
        return 1
    return -1


fn = np.vectorize(f)

g = Sigmoid(1)

nl = NeuronLayer(g, 2, 1)
nl.weights = np.array([[1], [0]])
print(nl.compute(np.array([5, 5])))
print(nl.activation_levels)


net = Network(np.array([2, 2, 1]), np.array([g, g]))
print(nl.compute(np.array([-5, 5])))
