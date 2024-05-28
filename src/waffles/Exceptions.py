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