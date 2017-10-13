import numpy as np
from function.function import Function


class Tanh(Function):
    """
    @brief      Classe définissant une tangeante hyperbolique formelle
    """

    def __init__(self, k=1, alpha=1):
        self.k = k
        self.alpha = alpha

    def out(self):
        return lambda x: self.k*np.tanh(self.alpha*x)

    def derivate(self):
        return lambda x: self.k*self.alpha/(np.power(np.cosh(self.alpha*x), 2))
