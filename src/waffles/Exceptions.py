def GenerateExceptionMessage(
    code,
    issuer,
    reason=''
):
    """
    Parameters
    ----------
    code : int
    issuer : str
    reason : str

    Returns
    -----------
    str
    """

    message = f"{issuer} raised exception #{code}"

    if reason != '':
        message += f": {reason}"

    return message


def handle_missing_data(func):
    """This is a decorator which is meant to decorate
    the initialiser method (__init__) of any class
    which derives from WfAna. It is meant to catch
    the KeyError exception raised when there is
    some missing data in the provided input-parameters
    dictionary, and re-word the exception to inform
    the user about this.
    """

    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except KeyError as e:
            raise KeyError(GenerateExceptionMessage(
                1,
                'handle_missing_data()',
                "You are trying to instantiate/check a "
                "WfAna-derived class without providing the required"
                f" input parameters. {str(e)[1:-1]}"))
        
    return wrapper

class WafflesBaseException(Exception):
    """Exception raised when a Waffles-related error occurs.
    Waffles custom exceptions should derive from this class."""
    pass

class NoDataInFile(WafflesBaseException):
    """Exception raised when the file to be read is empty, 
    or it is not empty but there is no data of the expected 
    type (self-trigger or full-stream) in it."""
    pass

class IllFormedAnalysisFolder(WafflesBaseException):
    """Exception raised when the folder from which an
    analysis is run is ill-formed. P.e. it does not contain
    the minimal required files or folders."""
    pass

class IllFormedSteeringFile(WafflesBaseException):
    """Exception raised when the specified steering file
    for an analysis-run does not follow the required
    structure.
    """
    pass

class IllFormedParametersFile(WafflesBaseException):
    """Exception raised when the specified parameters
    file for an analysis-run does not follow the
    required structure.
    """
    pass

class IllFormedAnalysisClass(WafflesBaseException):
    """Exception raised when the analysis class to be run
    is ill-formed."""
    pass

class IncompatibleInput(WafflesBaseException):
    """Exception raised when the given input parameters
    are not compatible among each other. P.e. happens
    if some of the simultaneously defined parameters
    are mutually exclusive."""
    pass

class NonExistentDirectory(WafflesBaseException):
    """Exception raised when an specified directory does
    not exist."""