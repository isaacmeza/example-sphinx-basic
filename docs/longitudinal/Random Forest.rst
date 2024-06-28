Random Forest
=============

.. _random-forests:
Let

.. math::

    \operatorname{Oracle}_{\mathcal{F},\text{reg}}\left(\{z_i\},\{u_i\}\right) &= \operatorname{argmin}_{f\in\mathcal{F}}\frac{1}{n}\sum^n_{i=1}\left(u_i-f(z_i)\right)^2 \\
    \operatorname{Oracle}_{\mathcal{F},\text{class}}\left(\{x_i\},\{v_i\}, \{w_i\}\right) &= \operatorname{argmax}_{f\in\mathcal{F}}\frac{1}{n}\sum^n_{i=1} w_i \Pr_{Z_i\sim\operatorname{Ber}\left(\frac{1+f(x_i)}{2}\right)}\left(Z_i = v_i \right)

be oracles for the regression and (weighted) classification problems. For data :math:`A = \{a_1,\ldots, a_n\}`, define :math:`\mathcal{F}_A = \left\{\left(f\left(a_1\right), \ldots, f\left(a_n\right)\right): f \in \mathcal{F}\right\}`.

Estimator 1
===========

Whenever the function classes :math:`\mathcal{G}`, :math:`\mathcal{F'}` are already norm constrained, the estimator \ref{estimator:npiv_general} can be reduced to

.. math::

    \hat{g} = \arg \min_{g\in\mathcal{G}} 
    \max_{f' \in \mathcal{F'}} \mathbb{E}_n\left[2\left\{g(A)-Y\right\} f'(C')-f'(C')^2\right]

**Ensemble solution**

Consider the algorithm where for :math:`t=1, \ldots, T`:

.. math::

    \begin{aligned}
    & u_i^t=\left(y_i-\frac{1}{t-1} \sum_{\tau=1}^{t-1} g_\tau\left(a_i\right)\right), \quad f'_t=\operatorname{Oracle}_{\mathcal{F'}, \text{reg}}\left(\{c_i'\}, \{u_i^t\}\right) \\
    & v_i^t=1\left\{f'_t\left(c_i'\right)>0\right\} , w_i^t=\left|f'_t\left(c_i'\right)\right| \quad g_t=\operatorname{Oracle}_{\mathcal{G}, \text{class}}\left(\{a_i\}, \{v_i^t\}, \{w_i^t\}\right) \\
    &
    \end{aligned}

Suppose that the set :math:`\mathcal{F'}_{C'}` is a convex set. Then the ensemble :math:`\bar{g}=\frac{1}{T} \sum_{t=1}^T g_t`, is a :math:`O\left(\frac{\log (T)+1}{T}\right)`-approximate solution to the minimax problem.
