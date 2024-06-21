
Nested Nonparametric Instrumental Variable Regression
=====================================================

.. image:: https://readthedocs.org/projects/testingnn/badge/?version=latest
    :target: https://testingnn.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Overview
--------

This package aims to solve or estimate nonparametrically nested moment conditions. We analyze the closed form or approximate solutions under different function classes for the following estimators:

Estimators
----------

Sequential Nested NPIV
~~~~~~~~~~~~~~~~~~~~~~

Given observations :math:`(A_i,B_i,C_i)` in \tr, an initial estimator :math:`\hat{g}` which may be estimated in \tr, and hyperparameter values :math:`(\lambda,\mu)`, estimate

.. math::

   \hat{h} = \argmin_{h \in \mathcal{H}} \left[ \sup_{f \in \mathcal{F}} \left\{ 2 \cdot \textsc{loss}(f,\hat{g},h) - \textsc{penalty}(f,\lambda) \right\} + \textsc{penalty}(h,\mu) \right]

where 

.. math::

   \textsc{penalty}(f,\lambda) = \mathbb{E}_m\{f(C)^2\} + \lambda \cdot \|f\|^2_{\mathcal{F}}, \quad \textsc{penalty}(h,\mu) = \mu \cdot \|h\|^2_{\mathcal{H}}.


Sequential Nested NPIV: Ridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given observations :math:`(A_i,B_i,C_i)` in \tr, an initial estimator :math:`\hat{g}` which may be estimated in \tr, and a hyperparameter :math:`\mu`, estimate

.. math::

   \hat{h} = \argmin_{h \in \mathcal{H}} \left[ \sup_{f \in \mathcal{F}} \left\{ 2 \cdot \textsc{loss}(f,\hat{g},h) - \textsc{penalty}(f) \right\} + \textsc{penalty}(h,\mu) \right]

where 

.. math::

   \textsc{penalty}(f) = \mathbb{E}_m\{f(C)^2\}, \quad \textsc{penalty}(h,\mu) = \mu \cdot \mathbb{E}_m\{h(B)^2\}.


Closed Form Solutions for Sequential Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimator 1
^^^^^^^^^^^

We study the estimator:

.. math::

   \hat{g} = \argmin_{g \in \mathcal{G}} \max_{f' \in \mathcal{F'}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right] - \lambda \|f\|_{\mathcal{F}}^2 + \mu' \|g\|_{\mathcal{G}}^2

where A is the set of endogenous variables, C' the set of instruments.

Estimator 2
^^^^^^^^^^^

We study the estimator:

.. math::

   \hat{g} = \argmin_{g \in \mathcal{G}} \max_{f' \in \mathcal{F'}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right] + \mu' \mathbb{E}_n \{ g(A)^2 \}


Joint Estimator
---------------

Simultaneous Nested NPIV
~~~~~~~~~~~~~~~~~~~~~~~~

Given observations :math:`(A_i,B_i,C_i,C_i')` in \tr, and hyperparameter values :math:`(\mu',\mu)`, estimate

.. math::

   (\hat{g},\hat{h}) = \argmin_{g \in \mathcal{G}, h \in \mathcal{H}} \bigg[ \sup_{f' \in \mathcal{F}} \left\{ 2 \cdot \textsc{loss}(f',Y,g) - \textsc{penalty}(f') \right\} + \textsc{penalty}(g,\mu') 
   + \sup_{f \in \mathcal{F}} \left\{ 2 \cdot \textsc{loss}(f,g,h) - \textsc{penalty}(f) \right\} + \textsc{penalty}(h,\mu) \bigg]

using analogous \textsc{penalty} notation to Estimator :ref:`estimator:npiv_ridge`.

Closed Form Solution for Joint Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The joint estimator solves:

.. math::

   (\hat{g},\hat{h}) = \argmin_{g \in \mathcal{G}, h \in \mathcal{H}} \max_{f' \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right] + \mu' \mathbb{E}_n \{ g(A)^2 \}
   + \max_{f \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ h(B) - g(A) \right\} f(C) - f(C)^2 \right] + \mu \mathbb{E}_n \{ h(B)^2 \}


Implementation
--------------

This documentation implements longitudinal estimation of functions :math:`g` and :math:`h` for several function classes:

- RKHS
- Random Forest
- Neural Networks
- Sparse Linear
- Linear

This documentation will provide details on how each class is implemented and how to use the commands.

Moreover, for the estimation of a semiparametric model, we have implemented double machine learning based on the estimation of the nuisance :math:`g` and :math:`h`.


Example Project usage
---------------------

This project has a standard Sphinx layout which is built by Read the Docs almost the same way that you would build it locally (on your own laptop!).

You can build and view this documentation project locally - we recommend that you activate `a local Python virtual environment first <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment>`_:

.. code-block:: console

    # Install required Python dependencies (Sphinx etc.)
    pip install -r docs/requirements.txt

    # Enter the Sphinx project
    cd docs/
    
    # Run the raw sphinx-build command
    sphinx-build -M html . _build/


You can also build the documentation locally with ``make``:

.. code-block:: console

    # Enter the Sphinx project
    cd docs/
    
    # Build with make
    make html
    
    # Open with your preferred browser, pointing it to the documentation index page
    firefox _build/html/index.html
