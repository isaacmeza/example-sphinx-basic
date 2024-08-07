"""
This is a module for demonstrating Sphinx documentation.
"""

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.nonparametric.kde import kernel_switch
import warnings
from tqdm import tqdm  # Import tqdm
import copy
import torch
from mliv.rkhs import ApproxRKHSIVCV
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar

device = torch.cuda.current_device() if torch.cuda.is_available() else None

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


