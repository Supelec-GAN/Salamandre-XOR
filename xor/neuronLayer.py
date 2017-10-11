import numpy as np


class NeuronLayer:
    """Classe permettant de créer une couche de neurones"""

    def __init__(self, function, input_size=1, output_size=1, ):
        self._weights = np.random.randn(input_size, output_size)
        self._bias = np.zeros((1, output_size))
        self._activation_function = function.out()
        self.activation_levels = np.zeros(output_size)
        self.output = np.zeros(output_size)

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
        self.output = self._activation_function(self.activation_levels)
        return self.output

    def backprop(self, out_influence, eta, input_layer):
        weight_influence = self.calculate_weight_influence(input_layer, out_influence)
        self.update_weights(eta, weight_influence)
        bias_influence = self.calculate(out_influence)
        self.update_bias(eta, bias_influence)
        in_influence = self.derivate_error(out_influence)
        return in_influence

    def update_weights(self, eta, weight_influence):
        self._weights = self._weights + eta*weight_influence

    def update_bias(self, eta, bias_influence):
        self._bias = self._bias + eta*bias_influence

    def calculate_weight_influence(self, input_layer, out_influence):
        S = self.activation_levels
        g_prime = self._activation_function.derivate(S)
        return np.dot(np.dot(input_layer, g_prime), out_influence)

    def calculate_bias_influence(self, out_influence):
        S = self.activation_levels
        g_prime = self._activation_function.derivate(S)
        return -np.dot(g_prime, out_influence)

    def derivate_error(self, out_influence):
        WT = np.transpose(self._weights)
        S = self.activation_levels
        g_prime = self._activation_function.derivate(S)
        return np.dot(np.dot(WT, g_prime), out_influence)
