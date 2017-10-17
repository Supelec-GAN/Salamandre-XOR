import numpy as np
import matplotlib.pyplot as plt
from xor.neuronLayer import NeuronLayer
from xor.network import Network
from function.tanh import Tanh

# Set d'apprentissage pour le XOR, 4 points
pt1 = np.array([[-1], [-1]])
pt2 = np.array([[-1], [1]])
pt3 = np.array([[1], [1]])
pt4 = np.array([[1], [-1]])
pt = [pt1, pt2, pt3, pt4]

delta = 0.01
eta = 0.2
activation_functions = np.array([Tanh(1.7159, 2/3), Tanh(1.7159, 2/3)])
neurons_count = np.array([2, 2, 1])
net = Network(neurons_count, activation_functions)


def print_network(net):
    i = 0
    for layer in net._layers_list:
        print("couche n°", i)
        print(layer.weights)
        i += 1

print_network(net)
error_check = 1
iterations = 10000
error_evolution = np.array([])
iterations_done = np.array([])
iterations_left = iterations
training_batch_abs = (np.random.random(iterations_left)-0.5)*2
training_batch_ord = (np.random.random(iterations_left)-0.5)*2


while iterations_left > 0:
    iterations_done = np.concatenate((iterations_done,[iterations - iterations_left]))

    inputs = np.array([[training_batch_abs[iterations_left-1]],
                       [training_batch_ord[iterations_left-1]]])  # vecteur colonne
    output = net.compute(inputs)
    if training_batch_ord[iterations_left-1]*training_batch_abs[iterations_left-1] > 0:
        reference = -1.7159
    else:
        reference = +1.7159
    error_check = net.error(output, reference)
    error_evolution = np.concatenate((error_evolution,[error_check]))
    net.backprop(delta, eta, inputs, reference)
    iterations_left -= 1

plt.plot(iterations_done, error_evolution)
plt.xlabel('Itérations éffectuées')
plt.ylabel('erreur')
plt.title("Evolution de l'erreur en fonction du temps")
plt.show()

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
print_network(net)
print("error =", error_check)
print("il reste =", iterations_left)
print(net.compute(np.array(pt1)))
print(net.compute(np.array(pt2)))
print(net.compute(np.array(pt3)))
print(net.compute(np.array(pt4)))
