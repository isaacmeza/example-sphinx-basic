Estimators for Sequential and Simultaneous Nested NPIV
======================================================

In this document, we analyze the closed-form or approximate solutions under different function classes for the following estimators:

**Sequential Nested NPIV:**

Given observations :math:`(A_i, B_i, C_i)` in \tr, an initial estimator :math:`\hat{g}` which may be estimated in \tr, and hyperparameter values :math:`(\lambda, \mu)`, estimate 

.. math::
   \hat{h} = \arg\min_{h \in \mathcal{H}} \left[ \sup_{f \in \mathcal{F}} \left\{ 2 \cdot \textsc{loss}(f, \hat{g}, h) - \textsc{penalty}(f, \lambda) \right\} + \textsc{penalty}(h, \mu) \right]

where :math:`\textsc{penalty}(f, \lambda) = \mathbb{E}_m\{f(C)^2\} + \lambda \cdot \|f\|^2_{\mathcal{F}}` and :math:`\textsc{penalty}(h, \mu) = \mu \cdot \|h\|^2_{\mathcal{H}}`.

**Sequential Nested NPIV: Ridge:**

Given observations :math:`(A_i, B_i, C_i)` in \tr, an initial estimator :math:`\hat{g}` which may be estimated in \tr, and a hyperparameter :math:`\mu`, estimate 

.. math::
   \hat{h} = \arg\min_{h \in \mathcal{H}} \left[ \sup_{f \in \mathcal{F}} \left\{ 2 \cdot \textsc{loss}(f, \hat{g}, h) - \textsc{penalty}(f) \right\} + \textsc{penalty}(h, \mu) \right]

where :math:`\textsc{penalty}(f) = \mathbb{E}_m\{f(C)^2\}` and :math:`\textsc{penalty}(h, \mu) = \mu \cdot \mathbb{E}_m\{h(B)^2\}`.

**Simultaneous Nested NPIV:**

Given observations :math:`(A_i, B_i, C_i, C_i')` in \tr\, and hyperparameter values :math:`(\mu', \mu)`, estimate 

.. math::
   (\hat{g}, \hat{h}) = \arg\min_{g \in \mathcal{G}, h \in \mathcal{H}} \left[ \sup_{f' \in \mathcal{F}} \left\{ 2 \cdot \textsc{loss}(f', Y, g) - \textsc{penalty}(f') \right\} + \textsc{penalty}(g, \mu') + \sup_{f \in \mathcal{F}} \left\{ 2 \cdot \textsc{loss}(f, g, h) - \textsc{penalty}(f) \right\} + \textsc{penalty}(h, \mu) \right]

using analogous :math:`\textsc{penalty}` notation to the Sequential estimators.


.. toctree::
   :maxdepth: 2

   longitudinal/RKHS
   longitudinal/Random Forest
   longitudinal/Neural Network
   longitudinal/Sparse Linear
   longitudinal/Regularized Linear
   longitudinal/Linear



