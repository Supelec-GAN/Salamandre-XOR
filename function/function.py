class Function:
    """
    @brief      classe abstraite de fonction formelle
    """

    def __init__(self, *args):
        pass

    def out(self):
        return lambda x: x

    def derivate(self):
        return lambda x: 1
