"""
This is a module for demonstrating Sphinx documentation.
"""

class MyClass:
    """
    A simple example class.

    Attributes
    ----------
    attribute1 : str
        Description of attribute1.
    attribute2 : int
        Description of attribute2.
    """

    def __init__(self, attribute1, attribute2):
        """
        Initialize the MyClass instance.

        Parameters
        ----------
        attribute1 : str
            Description of attribute1.
        attribute2 : int
            Description of attribute2.
        """
        self.attribute1 = attribute1
        self.attribute2 = attribute2

    def my_method(self, param1, param2):
        """
        Example method in MyClass.

        Parameters
        ----------
        param1 : str
            Description of param1.
        param2 : int
            Description of param2.

        Returns
        -------
        bool
            Description of return value.
        """
        return True

def my_function(param1, param2):
    """
    Example standalone function.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int
        Description of param2.

    Returns
    -------
    bool
        Description of return value.
    """
    return True