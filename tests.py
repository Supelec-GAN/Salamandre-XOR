import numpy as np
from brain.network import Network
from fonction import Sigmoid, Tanh, XorTest
from interface import Interface
from run import Run
from dataProcessor import DataProcessor
from errorGraphs import ErrorGraphs
from dataInterface import DataInterface


# iterations = [1000, 5000, 10000, 50000, 100000]
# eta = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

iterations = [100]
eta = [0.02]
iterations_test = 3


activation_functions = np.array(
    [Tanh(1.7159, 2 / 3), Tanh(1.7159, 2 / 3), Tanh(1.7159, 2 / 3)])
neurons_count = np.array([2, 2, 2, 1])
parallel_learnings = 10
test_period = 10
batch_test = 2 * np.random.random_sample((iterations_test, 2)) - 1

net = Network(neurons_count, activation_functions)


data_interface = DataInterface('xor_result')


for i in range(len(iterations)):
        for j in range(len(eta)):
                batch = 2 * np.random.random_sample((iterations[i], 2)) - 1
                r = Run(net, XorTest(-1.7159, 1.7159), batch, batch_test, parallel_learnings, eta[j], test_period)
                tests_passed = r.learning_manager()[0]
                dp = DataProcessor(parallel_learnings, iterations, test_period)

                data_param = np.array([parallel_learnings, test_period, iterations[i], eta[j]])
                data_interface.save(tests_passed, 'tests_passed', data_param)

                error_bar = dp.error_bar(tests_passed)
                eg = ErrorGraphs(parallel_learnings, error_bar, eta, neurons_count, test_period)
                x, y = dp.test_data(batch, tests_passed)
                eg = ErrorGraphs(parallel_learnings, error_bar, eta, neurons_count, test_period)
                eg.error_graphs_test(x, y)
