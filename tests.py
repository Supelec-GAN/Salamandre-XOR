import numpy as np
from brain.network import Network
from fonction import Sigmoid, Tanh, XorTest, SigmoidCentered
from interface import Interface
from run import Run
from dataProcessor import DataProcessor
from errorGraphs import ErrorGraphs
from dataInterface import DataInterface


# iterations = [1000, 5000, 10000, 50000, 100000]
#eta = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

iterations = [10000]
eta = [0.2]
iterations_test = 1


activation_functions = np.array(
    [SigmoidCentered(1), SigmoidCentered(1), SigmoidCentered(1)])
neurons_count = np.array([2, 2, 1])
parallel_learnings = 1
test_period = 1000
batch_test = np.random.random_sample((iterations_test, 2))

net = Network(neurons_count, activation_functions)


data_interface = DataInterface('xor_result')

xor = XorTest(-1, 1)
interface = Interface(net, xor)

for i in range(len(iterations)):
        for j in range(len(eta)):
                batch = np.random.random_sample((iterations[i], 2))*(xor.maxi-xor.mini) + xor.mini
                r = Run(net, xor, batch, batch_test, parallel_learnings, eta[j], test_period)
                tests_passed = r.learning_manager()[0]
                dp = DataProcessor(parallel_learnings, iterations, test_period)

                data_param = np.array([parallel_learnings, test_period, iterations[i], eta[j]])
                data_interface.save(tests_passed, 'tests_passed', data_param)

                error_bar = dp.error_bar(tests_passed)
                eg = ErrorGraphs(parallel_learnings, error_bar, eta, neurons_count, test_period)
                x, y = dp.test_data(batch, tests_passed)
                eg.error_graphs_test(x, y)

                interface.print_grid_net(100)
                interface.error_graphs(x, y, test_period, parallel_learnings, error_bar, eta, neurons_count)