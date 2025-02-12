import copy

from typing import Optional, List, Any

from waffles.Exceptions import GenerateExceptionMessage


class Map:
    """This class implements a list of lists which make up a
    bi-dimensional non-ragged array which represents a map.
    Each entry of this non-ragged array is an object of a
    certain type, which is homogenous across the whole
    array.

    Attributes
    ----------
    rows: int
        The number of rows in the map
    columns: int
        The number of columns in the map
    type: type
        The type of the objects stored in the map
    data: list of lists
        Nested list which contains the data of the
        map. data[i][j] gives the object stored in
        the i-th row and j-th column of the map.

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(
        self: int,
        rows: int,
        columns: int,
        type_: type,
        data: Optional[List[List[Any]]] = None
    ):
        """Map class initializer

        Parameters
        ----------
        rows: int
            It must be a positive integer
        columns: int
            It must be a positive integer
        type_: type
        data: list of lists
            It must be a list of lists of objects of the
            same type as the one specified in the type_
            parameter. The length of data must be equal to
            rows and the length of each one of its lists
            must be equal to columns.
        """

        # Shall we add type checks here?

        if rows < 1:
            raise Exception(GenerateExceptionMessage(
                1,
                'Map.__init__()',
                f"The given number of rows ({rows}) must be positive."))
            
        if columns < 1:
            raise Exception(GenerateExceptionMessage(
                2,
                'Map.__init__()',
                f"The given number of columns ({columns}) must be positive."))

        if not Map.list_of_lists_is_well_formed(data, rows, columns):
            raise Exception(GenerateExceptionMessage(
                3,
                'Map.__init__()',
                f"The shape of the given data is not ({rows}, {columns})."))

        if not all([isinstance(item, type_) for row in data for item in row]):
            raise Exception(GenerateExceptionMessage(
                4,
                'Map.__init__()',
                f"The type of the objects in the given data must be {type_}."))
            
        self.__rows = rows
        self.__columns = columns
        self.__type = type_
        self.__data = data

    # Getters
    @property
    def rows(self):
        return self.__rows

    @property
    def columns(self):
        return self.__columns

    @property
    def type(self):
        return self.__type

    @property
    def data(self):
        return self.__data

    @staticmethod
    def list_of_lists_is_well_formed(
            grid: List[List[Any]],
            nrows: int,
            ncols: int
    ) -> bool:
        """This method returns True if the given grid contains
        nrows lists, each of which has a length equal to ncols.
        It returns False if else.

        Parameters
        ----------
        grid: list of lists
        nrows: int
        ncols: int

        Returns
        ----------
        bool
        """

        if nrows < 1 or ncols < 1:
            raise Exception(GenerateExceptionMessage(
                1,
                'Map.list_of_lists_is_well_formed()',
                'The number of rows and columns must be positive.'))
            
        if len(grid) != nrows:
            return False
        else:
            for row in grid:
                if len(row) != ncols:
                    return False
        return True

    @classmethod
    def from_unique_value(
        cls,
        nrows: int,
        ncols: int,
        type_: type,
        value,
        independent_copies=False
    ) -> 'Map':
        """This method returns a Map object whose rows, columns
        and type attributes match the input parameters nrows,
        ncols and type_, and for which all of its entries
        are equal to the input value.

        Parameters
        ----------
        nrows (resp. ncols): int
            Number of rows (resp. columns) of the returned
            Map object. It must be a positive integer.
        type_: type
            Type of the object(s) stored in the returned Map
            object
        value
            Its type must match the type_ parameter. It is
            the value that all the entries of the returned
            Map object will have.
        independent_copies: bool
            If True, the returned Map object will have
            independent copies of the value parameter in each
            one of its entries. If False, the returned Map object
            will have  references to the same object in each one
            of its entries.

        Returns
        ----------
        Map
        """

        if nrows < 1 or ncols < 1:
            raise Exception(GenerateExceptionMessage(
                1,
                'Map.from_unique_value()',
                'The number of rows and columns must be positive.'))
        if not isinstance(value, type_):
            raise Exception(GenerateExceptionMessage(
                2,
                'Map.from_unique_value()',
                'The type of the given value must match the given type.'))

        if not independent_copies:
            aux = [[value for _ in range(ncols)] for _ in range(nrows)]
        else:
            aux = [
                [copy.deepcopy(value) for _ in range(ncols)]
                for _ in range(nrows)]

        return cls(
            nrows,
            ncols,
            type_,
            data=aux)
