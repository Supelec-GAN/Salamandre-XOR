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
        # out_influence = self.derivate(delta, reference)
        if n == 1:
            input_layer = inputs
        else:
            input_layer = self._layers_list[-2].output

        out_influence = self._layers_list[n-1].init_derivate_error(reference)
        self._layers_list[n-1].update_weights(eta, self._layers_list[n-1].calculate_weight_influence(input_layer, out_influence))
        self._layers_list[n-1].update_bias(eta, self._layers_list[n-1].calculate_bias_influence(out_influence))

        for i in range(n-2, 0, -1):
            input_layer = self._layers_list[i - 1].output
            # out_influence = self._layers_list[i].backprop(out_influence, eta, input_layer, self._layers_list[i+1].weights)
            out_influence = self._layers_list[i].derivate_error(out_influence, self._layers_list[i+1].weights)
            self._layers_list[i].update_weights(eta, self._layers_list[i].calculate_weight_influence(input_layer, out_influence))
            self._layers_list[i].update_bias(eta, self._layers_list[i].calculate_bias_influence(out_influence))

        if n > 1:
            input_layer = inputs

            out_influence = self._layers_list[0].derivate_error(out_influence, self._layers_list[1].weights)
            self._layers_list[0].update_weights(eta, self._layers_list[0].calculate_weight_influence(input_layer, out_influence))
            self._layers_list[0].update_bias(eta, self._layers_list[0].calculate_bias_influence(out_influence))

    def derivate(self, delta, reference):
        test = self._layers_list[-1].init_derivate_error(reference)
        test2 = (self.error(self.output, reference) - self.error(self.output + delta, reference))/delta
        # print(test, test2)
        return test
        # return -2*self._layers_list[-1]._activation_function.derivate()(self._layers_list[-1]._activation_levels)
