import numpy as np
import matplotlib.pyplot as plt
from xor.network import Network
from function.tanh import Tanh

def fonction_test (input) : #Renvoie la réference attendue, celle si est pour le XOR
    if input[0]*input[1] > 0 :
        return 1
    else :
        return -1


def print_grid_net(net, grid):
    ab = np.linspace(-1, 1, grid)
    od = np.linspace(-1, 1, grid)
    abs_affiche = []
    ord_affiche = []

    for a in ab:
        for o in od:
            if net.compute(np.array([[a], [o]])) > 0:
                abs_affiche.append(a)
                ord_affiche.append(o)

    plt.plot(abs_affiche, ord_affiche, 'o')
    plt.axis([-1, 1, -1, 1])
    plt.grid()
    plt.show()


def learning_manager (batch, parrallel_learnings, activation_functions, neurons_count, iterations, eta) :
    iterations_left = iterations
    errors = np.zeros((iterations, parrallel_learnings))
    output_list = np.zeros((iterations, parrallel_learnings))
    reference_list = np.zeros((iterations, parrallel_learnings))

    for i in range(parrallel_learnings):
        net = Network(neurons_count, activation_functions)

        while iterations_left > 0 and errors[i][-1] > 0.01 :
            output = net.compute(batch[iterations-iterations_left])
            output_list[iterations-iterations_left][i] = output
            reference = fonction_test(batch[iterations-iterations_left])
            reference_list[iterations-iterations_left][i] = reference
            errors[iterations-iterations_left][i] = net.error(output,reference)
            net.backprop(eta, batch[iterations-iterations_left], reference)
            iterations_left -= 1

    return errors


def error_graphs(errors, iterations, parallel_learnings):
    mean_error = np.mean(errors, axis = 1)
    plt.plot(range(iterations), mean_error)
    plt.xlabel('Itérations')
    plt.ylabel('Erreur moyenne sur'+parrallel_learnings + "apprentissages")
    plt.title ("Evolution de l'erreur au fur et a mesure des apprentissages")