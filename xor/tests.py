import numpy as np
from xor.network import Network
from function.tanh import Tanh


iterations = [1000, 5000, 10000, 50000, 100000]
eta = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

activation_functions = np.array([Tanh(1.7159, 2 / 3), Tanh(1.7159, 2 / 3), Tanh(1.7159, 2 / 3)])
neurons_count = np.array([2, 3, 2, 1])
parallel_learnings = 100

errors = np.zeros((len(iterations),len(eta)))


for i in range(len(iterations)):
        for j in range(len(eta)):
                batch = 2 * np.random.random_sample((iterations[i], 2)) - 1
                errors[i][j] = learning_manager(batch, parallel_learnings, activation_functions, neurons_count, iterations[i], eta[j])



file = open('resultats.py', 'w')
file.write(str(errors))
file.close