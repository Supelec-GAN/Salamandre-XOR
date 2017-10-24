import numpy as np
from xor.network import Network
from function.sigmoid import Sigmoid
from function.tanh import Tanh
import Interface as interf



#iterations = [1000, 5000, 10000, 50000, 100000]
#eta = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

iterations = [10000]
eta = [0.02]
iterations_test = 3000


activation_functions = np.array([Tanh(1.7159, 2/3), Tanh(1.7159, 2/3), Tanh(1.7159, 2/3)])
neurons_count = np.array([2, 3, 2, 1])
parallel_learnings = 5
test_period = 100



for i in range(len(iterations)):
        for j in range(len(eta)):
                batch = 2 * np.random.random_sample((iterations[i], 2)) - 1
                batch_test = 2 * np.random.random_sample((iterations_test, 2)) - 1
                errors = interf.learning_manager(batch, batch_test, parallel_learnings, activation_functions, neurons_count, eta[j], test_period)
                errors_during_learning = errors[0]
                errors_during_test = errors[1]


