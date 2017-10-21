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


def learning_manager (batch, batch_test, parrallel_learnings, activation_functions, neurons_count, eta, test_period) :
    iterations = len(batch)
    iterations_test = len(batch_test)
    iterations_left = iterations
    errors_during_learning = np.zeros((iterations, parrallel_learnings))
    errors_during_test = np.zeros((iterations_test, parrallel_learnings))
    mean_error_during_test = np.rezos((iterations//test_period, parrallel_learnings))
    output_list_learning = np.zeros((iterations, parrallel_learnings))
    output_list_test = np.zeros((iterations_test, parrallel_learnings))
    reference_list_learning = np.zeros((iterations, parrallel_learnings))
    reference_list_test = np.zeros((iterations_test, parrallel_learnings))

    for i in range(parrallel_learnings):
        net = Network(neurons_count, activation_functions)

        while iterations_left > 0 and errors_during_learning[i][-1] > 0.01 :
            output = net.compute(batch[iterations-iterations_left])
            output_list_learning[iterations-iterations_left][i] = output
            reference = fonction_test(batch[iterations-iterations_left])
            reference_list_learning[iterations-iterations_left][i] = reference
            errors_during_learning[iterations-iterations_left][i] = net.error(output,reference)
            net.backprop(eta, batch[iterations-iterations_left], reference)
            iterations_left -= 1

            if iterations_left % test_period == 0 :
                for k in range(len(batch_test)) :
                    output = net.compute(batch_test[k])
                    output_list_test[k][i] = output
                    reference = fonction_test(batch_test[k])
                    reference_list_test[k][i] = reference
                    errors_during_test[k][i] = net.error(output,reference)
                mean_error_during_test[iterations_left//test_period][i] = np.mean(errors_during_test, axis = 0)


plt.plot(range(len(batch)//test_period),[mean_error_during_test[k][-1] for k in range(len(mean_error_during_test))])
plt.ylabel("Erreur moyenne sur le batch de test" )
plt.xlabel ("Occurences des tests")
plt.title("Evolution de l'erreur effectuée toutes les"+test_period+"itérations d'apprentissage")

mean_error = np.mean(errors_during_learning, axis = 1)
plt.plot(range(iterations), mean_error)
plt.xlabel('Itérations')
plt.ylabel('Erreur moyenne sur'+parrallel_learnings + "apprentissages")
plt.title ("Evolution de l'erreur au fur et a mesure des apprentissages")