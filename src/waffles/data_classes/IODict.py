class IoDict(dict):
    """Stands for Input/Output Dictionary. This class
    inherits from the built-in dict class. This class
    adds no further members to the built-in dict, except
    for the fact that it re-implements the representation
    (__repr__) method.
    """

    def __repr__(self):
        """This re-implementation of the __repr__ method
        provides a line-per-item representation of the
        dictionary, showing the key, the type of the
        key, the value and the type of the value for
        each key-value pair in this dictionary.
        """

        aux = '{  '
        for key in self.keys():
            aux += f"\t {key} ({type(key)}) : {self[key]} ({type(self[key])}),\n"
        aux = aux[:-2] + '\t}'

        return aux
