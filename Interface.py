import numpy as np
import matplotlib.pyplot as plt
from xor.network import Network

def error_bar(array, parrallel_learnings):
    std = np.std(array)
    return (2*std/np.sqrt(parrallel_learnings))

def fonction_test(input):       # Renvoie la réference attendue, celle ci est pour le XOR
    if input[0]*input[1] > 0:
        return -1.7159
    else:
        return 1.7159


def error_graphs(abs_error_test, ord_error_test, abs_error_learning, ord_error_learning, test_period, parrallel_learnings):
    plt.figure()
    plt.errorbar(abs_error_test, ord_error_test,error_bar(ord_error_test, parrallel_learnings),None, fmt = '--o', ecolor = 'g', capthick=1)
    plt.ylabel("Erreur moyenne sur le batch de test pour les" + str(parrallel_learnings) + "apprentissages")
    plt.xlabel("Occurences des tests")
    plt.title("Evolution de l'erreur, test effectué tous les " + str(test_period) + "apprentissages")
    plt.show()

    #plt.plot(abs_error_learning, ord_error_learning, 'x')
    #plt.xlabel('Itérations')
    #plt.ylabel('Erreur moyenne sur ' + str(parrallel_learnings) + " apprentissages")
    #plt.title("Evolution de l'erreur au fur et a mesure des apprentissages")
    #plt.show()


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


def learning_manager(batch, batch_test, parallel_learnings, activation_functions, neurons_count, eta, test_period):
    iterations = len(batch)
    iterations_test = len(batch_test)
    errors_during_learning = np.zeros((iterations, parallel_learnings), dtype=np.ndarray)
    errors_during_test = np.zeros((iterations_test, parallel_learnings), dtype=np.ndarray)
    mean_error_during_test = np.zeros((iterations//test_period, parallel_learnings))
    mean_error_during_learning = np.zeros((iterations//100, parallel_learnings), dtype=np.ndarray)
    output_list_learning = np.zeros((iterations, parallel_learnings), dtype=np.ndarray)
    output_list_test = np.zeros((iterations_test, parallel_learnings), dtype=np.ndarray)
    reference_list_learning = np.zeros((iterations, parallel_learnings), dtype=np.ndarray)
    reference_list_test = np.zeros((iterations_test, parallel_learnings), dtype=np.ndarray)

    for i in range(parallel_learnings):
        net = Network(neurons_count, activation_functions)
        iterations_left = iterations

        while iterations_left > 1:
            output = net.compute(batch[iterations-iterations_left])
            output_list_learning[iterations-iterations_left][i] = output
            reference = fonction_test(batch[iterations-iterations_left])
            reference_list_learning[iterations-iterations_left][i] = reference
            errors_during_learning[iterations-iterations_left][i] = net.error(output, reference)
            if (iterations-iterations_left) % 100 == 0 and iterations != iterations_left :
                mean_error_during_learning[(iterations-iterations_left) // 100][i] = np.mean(errors_during_learning[iterations-iterations_left-100:iterations-iterations_left][i])
            net.backprop(eta, batch[iterations-iterations_left], reference)
            iterations_left -= 1

            if (iterations-iterations_left) % test_period == 0:
                for k in range(len(batch_test)):
                    output = net.compute(batch_test[k])
                    output_list_test[k][i] = output
                    reference = fonction_test(batch_test[k])
                    reference_list_test[k][i] = reference
                    errors_during_test[k][i] = net.error(output, reference)
                mean_error_during_test[(iterations-iterations_left)//test_period][i] = np.mean(errors_during_test[-100:][i])

    iteration_a_laquelle_batch_test = range(len(batch)//test_period)
    moyenne_erreur_sur_le_batch_test = np.mean(mean_error_during_test,1)
    iterations_effectuees = range(iterations // 100)
    moyenne_erreur_apprentissage = np.mean(mean_error_during_learning, axis=1)

    error_graphs(iteration_a_laquelle_batch_test, moyenne_erreur_sur_le_batch_test, iterations_effectuees,
                 moyenne_erreur_apprentissage, test_period, parallel_learnings)
    print(mean_error_during_learning)
    print_grid_net(net,100)

    return errors_during_learning, errors_during_test