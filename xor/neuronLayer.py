import numpy as np


class NeuronLayer:
    """Classe permettant de créer une couche de neurones"""

    def __init__(self, function, input_size=1, output_size=1):
        # Matrice de dimension q*p avec le nombre de sortie et p le nombre d'entrée
        self._input_size = input_size
        self._output_size = output_size
        self._weights = np.transpose(np.random.randn(input_size, output_size))
        self._bias = np.zeros((output_size, 1))            # Vecteur colonne
        self._activation_function = function
        self.activation_levels = np.zeros((output_size, 1))  # Vecteur colonne
        self.output = np.zeros((output_size, 1))             # Vecteur colonne

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

    ##
    # @brief      Calcul des sorties de la couche
    #
    # @param      inputs  Inputs

    def compute(self, inputs):
        self.activation_levels = np.dot(self._weights, inputs) - self._bias
        self.output = self._activation_function.out()(self.activation_levels)
        return self.output

    ##
    # @brief      Retropropagation au niveau d'une couche
    #
    # @param      out_influence  influence of output on the error
    # @param      eta            The eta
    # @param      input_layer    The input value of the layer
    #
    # @return     retourne influence of the input on the error
    #
    def backprop(self, out_influence, eta, input_layer):
        weight_influence = self.calculate_weight_influence(input_layer, out_influence)
        self.update_weights(eta, weight_influence)
        bias_influence = self.calculate_bias_influence(out_influence)
        self.update_bias(eta, bias_influence)
        in_influence = self.derivate_error(out_influence)
        return in_influence

    def update_weights(self, eta, weight_influence):
        self._weights = self._weights + eta*weight_influence

    def update_bias(self, eta, bias_influence):
        self._bias = self._bias + eta*bias_influence

    ##
    # @brief      Calculates the weight influence.
    #
    # @param      input_layer    input of the last compute
    # @param      out_influence  influence of output on the error
    #
    # @return     vecteur of same dimension than weights.
    #
    def calculate_weight_influence(self, input_layer, out_influence):
        activation = self.activation_levels
        deriv_vector = self._activation_function.derivate()(activation)
        n = np.size(self.activation_levels)
        # reshape pour np.diag
        deriv_diag = np.diag(np.reshape(deriv_vector, (n)))
        temp_matrice = np.dot(out_influence, np.transpose(input_layer))
        return np.dot(deriv_diag, temp_matrice)

    ##
    # @brief      Calculates the bias influence.
    #
    # @param      out_influence  influence of output on the error
    #
    # @return     vector of dimension of bias vector.
    #
    def calculate_bias_influence(self, out_influence):
        activation = self.activation_levels
        deriv_vector = self._activation_function.derivate()(activation)
        n = np.size(self.activation_levels)
        # reshape pour np.diag
        deriv_diag = np.diag(np.reshape(deriv_vector, (n)))
        return -np.dot(deriv_diag, out_influence)

    ##
    # @brief      Calculates the error derivation
    #
    # @param      out_influence  influence of output on the error
    #
    # @return     the error used in the recursive formula
    #
    def derivate_error(self, out_influence):
        activation = self.activation_levels
        deriv_vector = self._activation_function.derivate()(activation)
        n = np.size(self.activation_levels)
        # reshape pour np.diag
        deriv_diag = np.diag(np.reshape(deriv_vector, (n)))
        return np.dot(np.transpose(self._weights), np.dot(deriv_diag, out_influence))
