import numpy as np
from function.function import Function


class Sigmoid(Function):
    """
    @brief      Classe définissant une sigmoïde formelle
    """

    def __init__(self, mu=1):
        self.mu = mu

    def out(self):
        return lambda x: 1/(1+np.exp(-self.mu*x))

    def derivate(self):
        return lambda x: self.mu*np.exp(-self.mu*x)/(np.power(1+np.exp(-self.mu*x, 2)))
