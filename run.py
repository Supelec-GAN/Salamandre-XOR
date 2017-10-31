import numpy as np


class Run:

    def __init__(self, network, fonction_test):
        self.network = network
        self.fonction_test = fonction_test

    def learning_manager(self, batch, batch_test, parallel_learnings, eta, test_period):
        iterations = len(batch)
        iterations_test = len(batch_test)
        errors_during_learning = np.zeros(
            (iterations, parallel_learnings), dtype=np.ndarray)
        errors_during_test = np.zeros(
            (iterations_test, parallel_learnings), dtype=np.ndarray)
        mean_error_during_test = np.zeros(
            (iterations // test_period, parallel_learnings))
        mean_error_during_learning = np.zeros(
            (iterations // 100, parallel_learnings), dtype=np.ndarray)
        output_list_learning = np.zeros(
            (iterations, parallel_learnings), dtype=np.ndarray)
        output_list_test = np.zeros(
            (iterations_test, parallel_learnings), dtype=np.ndarray)
        reference_list_learning = np.zeros(
            (iterations, parallel_learnings), dtype=np.ndarray)
        reference_list_test = np.zeros(
            (iterations_test, parallel_learnings), dtype=np.ndarray)

        net = self.network

        for i in range(parallel_learnings):
            net.reset()
            iterations_left = iterations

            while iterations_left > 1:
                output = net.compute(batch[iterations - iterations_left])
                output_list_learning[iterations - iterations_left][i] = output
                reference = self.fonction_test.out()(
                    batch[iterations - iterations_left][0], batch[iterations - iterations_left][1])
                reference_list_learning[iterations -
                                        iterations_left][i] = reference
                errors_during_learning[iterations -
                                       iterations_left][i] = net.error(output, reference)
                if (iterations - iterations_left) % 100 == 0 and iterations != iterations_left:
                    mean_error_during_learning[(iterations - iterations_left) // 100][i] = np.mean(
                        errors_during_learning[iterations - iterations_left - 100:iterations - iterations_left][i])
                net.backprop(
                    eta, batch[iterations - iterations_left], reference)
                iterations_left -= 1

                if (iterations - iterations_left) % test_period == 0:
                    for k in range(len(batch_test)):
                        output = net.compute(batch_test[k])
                        output_list_test[k][i] = output
                        reference = self.fonction_test.out()(batch_test[k][0], batch_test[k][1])
                        reference_list_test[k][i] = reference
                        errors_during_test[k][i] = net.error(output, reference)
                    mean_error_during_test[(
                        iterations - iterations_left) // test_period][i] = np.mean(errors_during_test[-100:][i])

        return errors_during_learning, errors_during_test
