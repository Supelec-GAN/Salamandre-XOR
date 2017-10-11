import numpy as np
from xor.neuronLayer import *


class Network:
    """Classe permettant de créer un perceptron multicouche"""

    def __init__(self, layers_neuron_count, layers_activation_function):
        self._layers_count = np.size(layers_neuron_count) - 1
        self._layers_list = np.array(self._layers_count * [NeuronLayer()])
        for i in range(0, self._layers_count):
            self._layers_list[i] = NeuronLayer(layers_neuron_count[i],
                                               layers_neuron_count[i+1],
                                               layers_activation_function[i]
                                               )
        self.output = np.zeros(layers_neuron_count[-1])

    def compute(self, inputs):
        self._layers_list[0].compute(inputs)
        for i in range(1, self._layers_count):
            self._layers_list[i].compute(self._layers_list[i-1].output)
        return self._layers_list[-1].output