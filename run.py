import numpy as np


class Run:

    def __init__(self, network, fonction_test, batch, batch_test, parallel_learnings, eta, test_period):
        self.network = network
        self.fonction_test = fonction_test
        self.batch = batch
        self.batch_test = batch_test
        self.parallel_learnings = parallel_learnings
        self.eta = eta
        self.test_period = test_period

        self.iterations = len(batch)
        self.iterations_test = len(batch_test)
        self.errors_during_learning = np.zeros(
            (self.iterations, parallel_learnings), dtype=np.ndarray)
        self.errors_during_test = np.zeros(
            (self.iterations_test, parallel_learnings), dtype=np.ndarray)
        self.mean_error_during_test = np.zeros(
            (self.iterations // test_period, parallel_learnings))
        self.mean_error_during_learning = np.zeros(
            (self.iterations // 100, parallel_learnings), dtype=np.ndarray)
        self.output_list_learning = np.zeros(
            (self.iterations, parallel_learnings), dtype=np.ndarray)
        self.output_list_test = np.zeros(
            (self.iterations_test, parallel_learnings), dtype=np.ndarray)
        self.reference_list_learning = np.zeros(
            (self.iterations, parallel_learnings), dtype=np.ndarray)
        self.reference_list_test = np.zeros(
            (self.iterations_test, parallel_learnings), dtype=np.ndarray)
        self.tests_passed = np.zeros(
            (self.iterations//test_period, parallel_learnings), dtype=np.ndarray)

    def test(self, i, iterations_left):
        for k in range(len(self.batch_test)):
            output = self.network.compute(self.batch_test[k])
            self.output_list_test[k][i] = output
            reference = self.fonction_test.out()(self.batch_test[k][0], self.batch_test[k][1])
            self.reference_list_test[k][i] = reference
            self.errors_during_test[k][i] = self.network.error(output, reference)
            if self.errors_during_test[k][i] > reference :
                self.tests_passed[(self.iterations - iterations_left) // self.test_period][i]+=1
        self.tests_passed[(self.iterations - iterations_left) // self.test_period][i] = self.tests_passed[(self.iterations - iterations_left) // self.test_period][i]/len(self.batch_test)
        #self.mean_error_during_test[(self.iterations - iterations_left) // self.test_period][i] = np.mean(
            #self.errors_during_test[-100:][i])
        checkout_test = self.mean_error_during_test[(self.iterations - iterations_left) // self.test_period][i] < 0.04

        return checkout_test

    def learning_manager(self):

        net = self.network

        for i in range(self.parallel_learnings):
            net.reset()
            iterations_left = self.iterations

            while iterations_left > 1 and not checkout_test:
                output = net.compute(self.batch[self.iterations - iterations_left])
                self.output_list_learning[self.iterations - iterations_left][i] = output
                reference = self.fonction_test.out()(
                    self.batch[self.iterations - iterations_left][0], self.batch[self.iterations - iterations_left][1])
                self.reference_list_learning[self.iterations - iterations_left][i] = reference
                self.errors_during_learning[self.iterations - iterations_left][i] = net.error(output, reference)
                if (self.iterations - iterations_left) % 100 == 0 and self.iterations != iterations_left:
                    self.mean_error_during_learning[(self.iterations - iterations_left) // 100][i] = np.mean(
                        self.errors_during_learning[self.iterations - iterations_left - 100:self.iterations - iterations_left][i])
                net.backprop(
                    self.eta, self.batch[self.iterations - iterations_left], reference)
                iterations_left -= 1

                if (self.iterations - iterations_left) % self.test_period == 0:
                    checkout_test = self.test(i, iterations_left)

        return self.errors_during_learning, self.tests_passed
