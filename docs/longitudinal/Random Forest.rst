Random Forest
=============

.. _random-forests:
Let

.. math::

    \operatorname{Oracle}_{\mathcal{F},\text{reg}}\left(\{z_i\},\{u_i\}\right) &= \operatorname{argmin}_{f\in\mathcal{F}}\frac{1}{n}\sum^n_{i=1}\left(u_i-f(z_i)\right)^2 \\
    \operatorname{Oracle}_{\mathcal{F},\text{class}}\left(\{x_i\},\{v_i\}, \{w_i\}\right) &= \operatorname{argmax}_{f\in\mathcal{F}}\frac{1}{n}\sum^n_{i=1} w_i \Pr_{Z_i\sim\operatorname{Ber}\left(\frac{1+f(x_i)}{2}\right)}\left(Z_i = v_i \right)

be oracles for the regression and (weighted) classification problems. For data :math:`A = \{a_1,\ldots, a_n\}`, define :math:`\mathcal{F}_A = \left\{\left(f\left(a_1\right), \ldots, f\left(a_n\right)\right): f \in \mathcal{F}\right\}`.

Estimator 1
-----------

Whenever the function classes :math:`\mathcal{G}`, :math:`\mathcal{F'}` are already norm constrained, the estimator can be reduced to

.. math::

    \hat{g} = \arg \min_{g\in\mathcal{G}} 
    \max_{f' \in \mathcal{F'}} \mathbb{E}_n\left[2\left\{g(A)-Y\right\} f'(C')-f'(C')^2\right]

.. admonition:: Ensemble solution

    Consider the algorithm where for :math:`t=1, \ldots, T`:

    .. math::

        \begin{aligned}
        & u_i^t=\left(y_i-\frac{1}{t-1} \sum_{\tau=1}^{t-1} g_\tau\left(a_i\right)\right), \quad f'_t=\operatorname{Oracle}_{\mathcal{F'}, \text{reg}}\left(\{c_i'\}, \{u_i^t\}\right) \\
        & v_i^t=1\left\{f'_t\left(c_i'\right)>0\right\} , w_i^t=\left|f'_t\left(c_i'\right)\right| \quad g_t=\operatorname{Oracle}_{\mathcal{G}, \text{class}}\left(\{a_i\}, \{v_i^t\}, \{w_i^t\}\right) \\
        &
        \end{aligned}

    Suppose that the set :math:`\mathcal{F'}_{C'}` is a convex set. Then the ensemble :math:`\bar{g}=\frac{1}{T} \sum_{t=1}^T g_t`, is a :math:`O\left(\frac{\log (T)+1}{T}\right)`-approximate solution to the minimax problem.

Estimator 2
-----------

For the estimator

.. math::

    \hat{g} = \arg \min_{g\in\mathcal{G}} 
    \max_{f' \in \mathcal{F'}} \mathbb{E}_n\left[2\left\{g(A)-Y\right\} f'(C')-f'(C')^2\right]+\mu'\E_n\{g(A)^2\}

.. admonition:: Ensemble solution

    Consider the algorithm where for :math:`t=1, \ldots, T`:

    .. math::

        \begin{aligned}
        & u_i^t=\left(y_i-\frac{1}{t-1} \sum_{\tau=1}^{t-1} g_\tau\left(a_i\right)\right), \quad f'_t=\operatorname{Oracle}_{\mathcal{F'}, \text{reg}}\left(\{c_i'\}, \{u_i^t\}\right) \\
        & v_i^t=\frac{1}{\mu' t}\sum_{\tau=1}^{t}f'_\tau(c_i'), \qquad \qquad \qquad g_t=\operatorname{Oracle}_{\mathcal{G}, \text{reg}}\left(\{a_i\}, \{v_i^t\}\right) \\
        &
        \end{aligned}

    Suppose that the sets :math:`\mathcal{F'}_{C'}`, :math:`\mathcal{G}_{A}` are convex. Then the ensemble: :math:`\bar{g}=\frac{1}{T} \sum_{t=1}^T g_t`, is a :math:`O\left(\frac{\log (T)+1}{T}\right)`-approximate solution to the minimax problem.


Estimator 3
-----------

For the joint estimator with ridge regularization

.. math::

    (\hat{g},\hat{h}) = \arg \min _{g\in\mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_n\left[2\left\{g(A)-Y\right\} f'(C')-f'(C')^2\right]
     +\mu'\E_n\{g(A)^2\} \\
    &\quad +
    \max_{f \in \mathcal{F}} \mathbb{E}_n\left[2\left\{h(B)-g(A)\right\} f(C)-f(C)^2\right]   
    +\mu\E_n\{h(B)^2\}

.. admonition:: Ensemble solution

    Consider the algorithm where for :math:`t=1, \ldots, T`:

    .. math::

        \begin{aligned}
        & u_i'^{t}=\left(y_i-\frac{1}{t-1} \sum_{\tau=1}^{t-1} g_\tau\left(a_i\right)\right), \quad u_i^t=\frac{1}{t-1} \sum_{\tau=1}^{t-1} \bigg(g_\tau\left(a_i\right)-h_\tau\left(b_i\right)\bigg)\\
        & f'_t=\operatorname{Oracle}_{\mathcal{F'}, \text{reg}}\left(\{c_i'\}, \{u_i'^t\}\right),\quad f_t=\operatorname{Oracle}_{\mathcal{F}, \text{reg}}\left(\{c_i\}, \{u_i^t\}\right) \\
        & v_i'^t=\frac{1}{\mu't}\sum_{\tau=1}^{t}\bigg(f'_\tau(c'_i)-f_\tau(c_i)\bigg), \quad  \qquad  v_i^t=\frac{1}{\mu t}\sum_{\tau=1}^{t}f_\tau(c_i)\\
        &g_t=\operatorname{Oracle}_{\mathcal{G}, \text{reg}}\left(\{a_i\}, \{v_i'^t\}\right),  \qquad   h_t=\operatorname{Oracle}_{\mathcal{H}, \text{reg}}\left(\{b_i\}, \{v_i^t\}\right) \\
        &
        \end{aligned}

    Suppose that the sets :math:`\mathcal{F'}_{C'}`, :math:`\mathcal{F}_{C}`, :math:`\mathcal{G}_{A}`, :math:`\mathcal{H}_{B}` are all convex sets. Then the ensembles: :math:`\bar{g}=\frac{1}{T} \sum_{t=1}^T g_t`, :math:`\bar{h}=\frac{1}{T} \sum_{t=1}^T h_t`, are a :math:`O\left(\frac{\log (T)+1}{T}\right)`-approximate solution to the minimax problem.

