class Function:
    """
    @brief      classe abstraite de fonction formelle
    """

    def __init__(self, *args):
        pass

    ##
    # @brief      retourne la fonction
    #
    # @param      self
    #
    # @return     une fonction de type lambda x:
    #
    def out(self):
        return lambda x: x

    ##
    # @brief      retourne la fonction dérivée
    #
    # @param      self  The object
    #
    # @return     la dérivée formelle selon
    #
    def derivate(self):
        return lambda x: 1
