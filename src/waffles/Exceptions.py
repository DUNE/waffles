def generate_exception_message( code, 
                                issuer,
                                reason=''):
    
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

        if reason!='':
            message += f": {reason}"
        
        return message

def handle_missing_data(func):

    """
    This is a decorator which is meant to decorate
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
        except KeyError:
            raise KeyError(generate_exception_message(  1,
                                                        'handle_missing_data()',
                                                        'You are trying to instantiate a WfAna-derived class without providing the required input parameters.'))
    return wrapper