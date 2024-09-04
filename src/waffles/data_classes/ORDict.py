from waffles.data_classes.IODict import IODict


class ORDict(IODict):
    """Stands for Output Result Dictionary. This class
    inherits from the IODict class, and it is intended
    to store a set of resulting parameters from an
    arbitrary computation. This class adds no further
    members to the IODict class and its base class
    (the built-in dictionary) except for the fact that
    it re-implements the subscription (__getitem__) method.
    """

    def __getitem__(self, key):
        """This re-implementation of the __getitem__ 
        method of the built-in dict class rewords the 
        KeyError message that is raised when a key is 
        not found in the dictionary.
        """

        try:
            return super().__getitem__(key)
        except KeyError:
            raise KeyError(
                "There is no output parameter"
                f" in the results dictionary called '{key}'. "
                "The list of available output parameters is:"
                f" {list(self.keys())}.")
