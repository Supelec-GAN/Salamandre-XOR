import numpy as np
import matplotlib.pyplot as plt
from xor.neuronLayer import NeuronLayer
from xor.network import Network
from function.sigmoid import Sigmoid


pt1 = np.array([[-1, -1]])
pt2 = np.array([[-1, 1]])
pt3 = np.array([[1, 1]])
pt4 = np.array([[1, -1]])
pt = [pt1, pt2, pt3, pt4]

activation_functions = np.array([Sigmoid(1)])
neurons_count = np.array([2, 1])
net = Network(neurons_count, activation_functions)

for layer in net._layers_list:
    print(layer.weights)

test = False
count = 500
while not test and count > 0:
    test = True
    order = np.random.permutation(4)
    for i in range(4):
        count -= 1
        output = net.compute(pt[order[i]])

        if order[i] == 0:
            reference = 0
        else:
            reference = 1

        if net.error(output, reference) > 0.25:
            test = False
        net.backprop(0.001, 0.2, pt[order[i]], reference)

print('après apprentissage')
for layer in net._layers_list:
    print(layer.weights)
