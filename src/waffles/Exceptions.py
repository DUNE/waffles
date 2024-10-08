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