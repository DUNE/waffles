from typing import Optional, List, Any

from src.waffles.Exceptions import generate_exception_message

class Map:

    """
    This class implements a list of lists which make up a 
    bi-dimensional non-ragged array which represents a map. 
    Each entry of this non-ragged array is an object of a 
    certain type, which is homogenous across the whole
    array.

    Attributes
    ----------
    Rows : int
        The number of rows in the map
    Columns : int
        The number of columns in the map
    Type : type
        The type of the objects stored in the map

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  rows,
                        columns,
                        type_,
                        data : Optional[List[List[Any]]] = None):
        
        """
        Map class initializer
        
        Parameters
        ----------
        rows : int
            It must be a positive integer
        columns : int
            It must be a positive integer
        type_ : type
            It must be a type
        """

        ## Shall we add type checks here?

        if rows < 1:
            raise Exception(generate_exception_message( 1,
                                                        'Map.__init__()',
                                                        f"The given number of rows ({rows}) must be positive."))
        if columns < 1:
            raise Exception(generate_exception_message( 2,
                                                        'Map.__init__()',
                                                        f"The given number of columns ({columns}) must be positive."))
        if not isinstance(type_, type):
            raise Exception(generate_exception_message( 3,
                                                        'Map.__init__()',
                                                        f"The given type ({type_}) must be a type."))
        self.__rows = rows
        self.__columns = columns
        self.__type = type_

    #Getters
    @property
    def Rows(self):
        return self.__rows
    
    @property
    def Columns(self):
        return self.__columns
    
    @property
    def Type(self):
        return self.__type

    @staticmethod
    def list_of_lists_is_well_formed(   grid : List[List[Any]],
                                        nrows : int,
                                        ncols : int) -> bool:
        
        """
        This method returns True if the given grid contains
        nrows lists, each of which has a length equal to ncols.
        It returns False if else.

        Parameters
        ----------
        grid : list of lists
        nrows : int
        ncols : int

        Returns
        ----------
        bool
        """

        if nrows < 1 or ncols < 1:
            raise Exception(generate_exception_message( 1,
                                                        'Map.list_of_lists_is_well_formed()',
                                                        'The number of rows and columns must be positive.'))
        if len(grid) != nrows:
            return False
        else:
            for row in grid:
                if len(row) != ncols:
                    return False
        return True