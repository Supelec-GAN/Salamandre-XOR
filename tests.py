import numpy as np
from xor.network import Network
from fonction import Sigmoid, Tanh, XorTest
from interface import Interface
from run import Run as r
from dataProcessor import DataProcessor
from errorGraphs import ErrorGraphs


# iterations = [1000, 5000, 10000, 50000, 100000]
# eta = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

iterations = [10000]
eta = [0.02]
iterations_test = 3000


activation_functions = np.array([Tanh(1.7159, 2/3), Tanh(1.7159, 2/3), Tanh(1.7159, 2/3)])
neurons_count = np.array([2, 2, 2, 1])
parallel_learnings = 100
test_period = 100
batch_test = 2 * np.random.random_sample((iterations_test, 2)) - 1

net = Network(neurons_count, activation_functions)


interf = Interface(net,  XorTest(-1.7159, 1.7159))

for i in range(len(iterations)):
        for j in range(len(eta)):
                batch = 2 * np.random.random_sample((iterations[i], 2)) - 1
                tests_passed = r.learning_manager()[0]
                dp = DataProcessor(parallel_learnings, iterations, test_period)
                error_bar = dp.error_bar(tests_passed)
                eg = ErrorGraphs(parallel_learnings, error_bar, eta, neurons_count, test_period)
                x,y = dp.test_data(batch, tests_passed)
                eg = ErrorGraphs(parallel_learnings,error_bar, eta, neurons_count, test_period)
                eg.error_graphs_test(x,y)

