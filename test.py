"""
Lumache - Python library for cooks and food lovers.

This is a Python docstring, we can use reStructuredText syntax here!

.. code-block:: python

    # Import lumache
    import lumache

    # Call its only function
    get_random_ingredients(kind=["cheeses"])
"""

__version__ = "0.1.0"




def get_random_ingredients(kind=None):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]
    """
    return ["shells", "gorgonzola", "parsley"]
