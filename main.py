import numpy as np
import matplotlib.pyplot as plt
from xor.neuronLayer import NeuronLayer
from xor.network import Network
from function.tanh import Tanh


pt1 = np.array([[-1, -1]])
pt2 = np.array([[-1, 1]])
pt3 = np.array([[1, 1]])
pt4 = np.array([[1, -1]])
pt = [pt1, pt2, pt3, pt4]

activation_functions = np.array([Tanh(2/3)])
neurons_count = np.array([2, 1])
net = Network(neurons_count, activation_functions)

for layer in net._layers_list:
    print(layer.weights)
error_check = 1
iterations_left = 100000
training_batch_abs = (np.random.random(iterations_left)-0.5)*2
training_batch_ord = (np.random.random(iterations_left)-0.5)*2


while iterations_left > 0:
    output = net.compute([[training_batch_abs[iterations_left-1], training_batch_ord[iterations_left- 1]]])
    if training_batch_ord[iterations_left-1]*training_batch_abs[iterations_left-1] > 0:
        reference = 1
    else:
        reference = 0
    error_check = net.error(output, reference)
    net.backprop(0.01, 0.2, [[training_batch_abs[iterations_left-1], training_batch_ord[iterations_left - 1]]], reference)
    iterations_left -= 1


# test = False
# count = 500
# while not test and count > 0:
#    test = True
#    order = np.random.permutation(4)
#    for i in range(4):
#        count -= 1
#        output = net.compute(pt[order[i]])
#
#        if order[i] == 0:
#            reference = 0
#        else:
#            reference = 1
#
#        if net.error(output, reference) > 0.25:
#            test = False
#        net.backprop(0.001, 0.2, pt[order[i]], reference)

print('après apprentissage')
for layer in net._layers_list:
    print(layer.weights)
print("error =", error_check)
print("il reste =", iterations_left)
print(net.compute(np.array([[1.0, 1.0]])))
print(net.compute(np.array([[-1.0, 1.0]])))
print(net.compute(np.array([[1.0, -1.0]])))
print(net.compute(np.array([[-1.0, -1.0]])))
