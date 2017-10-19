import numpy as np
import matplotlib.pyplot as plt
from xor.network import Network
from function.tanh import Tanh

# Set d'apprentissage pour le XOR, 4 points
pt1 = np.array([[-1], [-1]])
pt2 = np.array([[-1], [1]])
pt3 = np.array([[1], [1]])
pt4 = np.array([[1], [-1]])
pt = [pt1, pt2, pt3, pt4]

eta = 0.01
activation_functions = np.array(
    [Tanh(1.7159, 2 / 3), Tanh(1.7159, 2 / 3), Tanh(1.7159, 2 / 3)])
neurons_count = np.array([2, 2, 3, 1])
net = Network(neurons_count, activation_functions)


def print_network(net):
    i = 0
    for layer in net._layers_list:
        print("couche n°", i)
        print("poids : ")
        print(layer.weights)
        print("biais : ")
        print(layer.bias)
        i += 1
    print('\n')
    print('########################################################################')
    print('\n')


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


def print_error(reference_list, output_list, n):
    print("erreur globale :", net.error(
        output_list, reference_list) / len(output_list))
    print("dernière erreur :", net.error(output_list[-1], reference_list[-1]))
    print("erreur sur les n dernières valeurs :", net.error(
        output_list[-n - 1:-1], reference_list[-n - 1:-1]) / n)
    print("Details des n dernières valeurs")
    for i in range(0, n):
        print(output_list[-n + i], reference_list[-n - i],
              net.error(output_list[-n + i], reference_list[-n - i]))


print("Valeurs initiales")
print_network(net)
# print_grid_net(net, 50)
error_check = 1
nb_iteration = 10000
iterations_left = nb_iteration
training_batch_abs = (np.random.random(iterations_left) - 0.5) * 2
training_batch_ord = (np.random.random(iterations_left) - 0.5) * 2

reference_list = np.zeros(iterations_left)
output_list = np.zeros(iterations_left)

while iterations_left > 0:

    inputs = np.array([[training_batch_abs[iterations_left - 1]],
                       [training_batch_ord[iterations_left - 1]]])  # vecteur colonne
    output = net.compute(inputs)
    output_list[iterations_left - 1] = output
    if training_batch_ord[iterations_left - 1] * training_batch_abs[iterations_left - 1] > 0:
        reference = +1
    else:
        reference = -1

    reference_list[iterations_left - 1] = reference

    error_check = net.error(output, reference)
    net.backprop(eta, inputs, reference)
    iterations_left -= 1


print('après apprentissage')
print_network(net)
print_error(reference_list, output_list, 20)
print("il reste =", iterations_left)

print("1 attendu : ", net.compute(np.array(pt1)))
print("-1 attendu : ", net.compute(np.array(pt2)))
print("1 attendu : ", net.compute(np.array(pt3)))
print("-1 attendu : ", net.compute(np.array(pt4)))

print_grid_net(net, 100)

plt.plot([net.error(output_list[i], reference_list[i])
          for i in range(nb_iteration - 1, 0, -10)], 'o')
plt.xlabel("Itérations")
plt.ylabel("Erreur")
plt.title("Evolution de l'erreur pendant l'apprentissage")
plt.show()
