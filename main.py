import numpy as np
import matplotlib.pyplot as plt
from xor.network import Network
from function.tanh import Tanh


batch = 2 * np.random.random_sample((iterations, 2)) - 1

# Set d'apprentissage pour le XOR, 4 points
pt1 = np.array([[-1], [-1]])
pt2 = np.array([[-1], [1]])
pt3 = np.array([[1], [1]])
pt4 = np.array([[1], [-1]])
pt = [pt1, pt2, pt3, pt4]

eta = 0.01
activation_functions = np.array(
    [Tanh(1.7159, 2 / 3), Tanh(1.7159, 2 / 3), Tanh(1.7159, 2 / 3)])
neurons_count = np.array([2, 3, 2, 1])
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
iterations = 10000
training_batch_abs = (np.random.random(iterations_left) - 0.5) * 2
training_batch_ord = (np.random.random(iterations_left) - 0.5) * 2




print('après apprentissage')
print_network(net)
print_error(reference_list, output_list, 20)
print("il reste =", iterations_left)


print_grid_net(net, 100)

plt.plot([net.error(output_list[i], reference_list[i])
          for i in range(nb_iteration - 1, 0, -10)], 'o')
plt.xlabel("Itérations")
plt.ylabel("Erreur")
plt.title("Evolution de l'erreur pendant l'apprentissage")
plt.show()
