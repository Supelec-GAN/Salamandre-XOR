import numpy as np
from xor.neuronLayer import NeuronLayer


class Network:
    """Classe permettant de créer un perceptron multicouche"""

    ##
    # @brief      Constructs the object.
    #
    # @param      self                        The object
    # @param      layers_neuron_count         Nombre de Neurones par couches,
    #                                         en incluant le nombres d'entrées en position 0
    # @param      layers_activation_function  The layers activation function
    #
    def __init__(self, layers_neuron_count, layers_activation_function):
        self._layers_count = np.size(layers_neuron_count) - 1
        self._layers_list = np.array(self._layers_count * [NeuronLayer(layers_activation_function[0])])
        for i in range(0, self._layers_count):
            self._layers_list[i] = NeuronLayer(layers_activation_function[i],
                                               layers_neuron_count[i],
                                               layers_neuron_count[i + 1]
                                               )
        self.output = np.zeros(layers_neuron_count[-1])

    ##
    # @brief      On calcule la sortie du réseau
    #
    # @param      self    The object
    # @param      inputs  The inputs
    #
    # @return     La sortie de la dernière couche est la sortie finale
    #
    def compute(self, inputs):
        self._layers_list[0].compute(inputs)
        for i in range(1, self._layers_count):
            self._layers_list[i].compute(self._layers_list[i - 1].output)
        return self._layers_list[-1].output

    ##
    # @brief      Calcul d'erreur quadratique
    #
    # @param      x  la sortie à comparer
    # @param      reference  The reference
    #
    # @return     norme2 de la différence de vecteur
    #
    def error(self, x, reference):
        return np.linalg.norm(x - reference)

    def backprop(self, delta, eta, inputs, reference):
        n = self._layers_count
        out_influence = self.derivate(delta, reference)
        if n == 1:
            self._layers_list[0].backprop(out_influence, eta, inputs)
            return 0
        else:
            for i in range(n-1, 0, -1):
                input_layer = self._layers_list[i - 1].output
                out_influence = self._layers_list[i].backprop(out_influence, eta, input_layer)

            self._layers_list[0].backprop(out_influence, eta, inputs)

    def derivate(self, delta, reference):
        return (self.error(self.output, reference) - self.error(self.output + delta, reference))/delta
