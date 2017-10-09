import numpy as np


class NeuronLayer:
    """Classe permettant de créer une couche de neurones"""

    def __init__(self, input_size, output_size, activation_function):
        self._weights = np.random.randn(input_size, output_size)
        self._bias = np.zeros((1, output_size))
        self._activation_function = activation_function
        self.activation_levels = np.zeros(output_size)

    @property
    def weights(self):
        """Get the current weights."""
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        self._weights = new_weights

    @weights.deleter
    def weights(self):
        del self._weights

    def compute(self, inputs):
        self.activation_levels = np.dot(inputs, self._weights) - self._bias
        return self._activation_function(self.activation_levels)
