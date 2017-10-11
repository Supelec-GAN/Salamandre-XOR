import numpy as np
import matplotlib.pyplot as plt
from xor.neuronLayer import NeuronLayer
from xor.network import Network
from function.sigmoid import Sigmoid


pt1 = np.array([-1, -1])
pt2 = np.array([-1, 1])
pt3 = np.array([1, 1])
pt4 = np.array([1, -1])
pt = [pt1, pt2, pt3, pt4]

activation_functions = np.array([Sigmoid(1)])
neurons_count = np.array([2, 1])
net = Network(neurons_count, activation_functions)



test = False
while not test:
    test = True
    order = np.random.permutation(4)
    for i in range(4):
        output = net.compute(pt[order[i]], activation_functions)

        if order[i] == 0:
            reference = -1
        else:
            reference = 1

        if net.error(output, reference) > 0.25:
            test = False
        net.backprop(0.001, 0.2, pt[order[i]], reference)

