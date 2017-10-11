import numpy as np
from xor.neuronLayer import NeuronLayer


class Network:
    """Classe permettant de créer un perceptron multicouche"""

    def __init__(self, layers_neuron_count, layers_activation_function):
        self._layers_count = np.size(layers_neuron_count) - 1
        self._layers_list = np.array(self._layers_count * [NeuronLayer(layers_activation_function[0])])
        for i in range(0, self._layers_count):
            self._layers_list[i] = NeuronLayer(layers_activation_function[i],
                                               layers_neuron_count[i],
                                               layers_neuron_count[i + 1]
                                               )
        self.output = np.zeros(layers_neuron_count[-1])

    def compute(self, inputs):
        self._layers_list[0].compute(inputs)
        for i in range(1, self._layers_count):
            self._layers_list[i].compute(self._layers_list[i - 1].output)
        return self._layers_list[-1].output

    def backprop(self, delta, eta, inputs, reference):
        n = self._layers_count
        out_influence = self.derivate(delta, reference)
        input_layer = self._layers_list[n - 1].output
        out_influence = self._layers_list[n].backprop(out_influence, eta, input_layer)

        for i in range(n - 1, 1, -1):
            input_layer = self._layers_list[i - 1].output
            out_influence = self._layers_list[i].backprop(out_influence, eta, input_layer)

        self._layers_list[0].backprop(out_influence, eta, inputs)

    def derivate(self, delta, reference):
        return (self.error(self.output, reference) - self.error(self.output + delta, reference))/delta