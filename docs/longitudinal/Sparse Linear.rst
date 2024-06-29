.. _sparse-linear-function-spaces:

Sparse Linear Function Spaces (:math:`\ell_1-\ell_1`)
=====================================================

In this section we address the high-dimensional case, where the function class is sparse linear, i.e. :math:`g(a) = \langle \alpha, a\rangle`, where :math:`\|\alpha\|_0 := \{j\in [p]\,|\,|\alpha_j|>0\} \leq s`. We will consider :math:`\ell_1` relaxations for the minimax optimization problem with :math:`\ell_1`-balls for the adversary. We remove the non-smoothness of the :math:`\ell_1` regularization by lifting the parameter :math:`\alpha` to a :math:`2p`-dimensional positive orthant. Consider two vectors :math:`\rho^{+}, \rho^{-} \geq 0` and then setting :math:`\alpha = \rho^{+} - \rho^{-}`, with :math:`\rho = \left(\rho^{+}; \rho^{-}\right)`. Observe that for any feasible :math:`\bar{\alpha}`, the solution :math:`\rho_i^{+} = \alpha_i 1\left\{\alpha_i > 0\right\}` and :math:`\rho_i^{-} = \alpha_i 1\left\{\alpha_i \leq 0\right\}` is still feasible and achieves the same objective, by the linearity of the loss function. Moreover, any solution :math:`\rho`, maps to a feasible solution :math:`\alpha` and thus the two optimization programs have the same optimal solutions.

Thus we will be solving an optimization problem over the :math:`2p`-dimensional simplex, and we will be using *Optimistic-Follow-the-Regularized-Leader* to find an :math:`\epsilon`-approximate solution. The approximate solutions of the minimax problems for all of our estimator will rely on the following proposition:

.. admonition:: Proposition 17 in `Dikkala et al. (2020) <https://arxiv.org/abs/2006.07201>`_
    :class: lemma
    :name: proposition-17

    Consider a minimax objective: :math:`\min _{\theta \in \Theta} \max _{w \in W} \ell(\theta, w)`. Suppose that :math:`\Theta, W` are convex sets and that :math:`\ell(\theta, w)` is convex in :math:`\theta` for every :math:`w` and concave in :math:`w` for any :math:`\theta`. Let :math:`\|\cdot\|_{\Theta}` and :math:`\|\cdot\|_W` be arbitrary norms in the corresponding spaces. Moreover, suppose that the following Lipschitzness properties are satisfied:

    .. math::

        \begin{aligned}
        & \forall \theta \in \Theta, w, w^{\prime} \in W: \left\|\nabla_\theta \ell(\theta, w) - \nabla_\theta \ell\left(\theta, w^{\prime}\right)\right\|_{\Theta, *} \leq L\left\|w - w^{\prime}\right\|_W \\
        & \forall w \in W, \theta, \theta^{\prime} \in \Theta: \left\|\nabla_w \ell(\theta, w) - \nabla_w \ell\left(\theta^{\prime}, w\right)\right\|_{W, *} \leq L\left\|\theta - \theta^{\prime}\right\|_W
        \end{aligned}

    where :math:`\|\cdot\|_{\Theta, *}` and :math:`\|\cdot\|_{W, *}` correspond to the dual norms of :math:`\|\cdot\|_{\Theta}` and :math:`\|\cdot\|_W`. Consider the algorithm where at each iteration each player updates their strategy based on:

    .. math::

        \begin{aligned}
        & \theta_{t+1} = \underset{\theta \in \Theta}{\arg \min } \theta^{\top}\left(\sum_{\tau \leq t} \nabla_\theta \ell\left(\theta_\tau, w_\tau\right) + \nabla_\theta \ell\left(\theta_t, w_t\right)\right) + \frac{1}{\eta} R_{\min }(\theta) \\
        & w_{t+1} = \underset{w \in W}{\arg \max } w^{\top}\left(\sum_{\tau \leq t} \nabla_w \ell\left(\theta_\tau, w_\tau\right) + \nabla_w \ell\left(\theta_t, w_t\right)\right) - \frac{1}{\eta} R_{\max }(w)
        \end{aligned}

    such that :math:`R_{\min }` is 1-strongly convex in the set :math:`\Theta` with respect to norm :math:`\|\cdot\|_{\Theta}` and :math:`R_{\max }` is 1-strongly convex in the set :math:`W` with respect to norm :math:`\|\cdot\|_W` and with any step-size :math:`\eta \leq \frac{1}{4 L}`. Then the parameters :math:`\bar{\theta} = \frac{1}{T} \sum_{t=1}^T \theta_t` and :math:`\bar{w} = \frac{1}{T} \sum_{t=1}^T w_t` correspond to an :math:`\frac{2 R_*}{\eta \cdot T}`-approximate equilibrium and hence :math:`\bar{\theta}` is a :math:`\frac{4 R_*}{\eta T}`-approximate solution to the minimax objective, where :math:`R` is defined as:

    .. math::

        R_* := \max \left\{\sup _{\theta \in \Theta} R_{\min }(\theta) - \inf _{\theta \in \Theta} R_{\min }(\theta), \sup _{w \in W} R_{\max }(w) - \inf _{w \in W} R_{\max }(w)\right\}


.. _estimator-1:

Estimator 1
-----------

The minimax problem is:

.. math::
    :label: minimax-sparse-est1

    \min_{\|\alpha\|_1 \leq V_1} \max _{\|\theta_1\|_1 \leq 1} L(\alpha, \theta) := \min_{\|\alpha\|_1 \leq V_1} \max _{\|\theta_1\|_1 \leq 1} 2\langle \mathbb{E}_n [(y - \langle \alpha, a \rangle)c'], \theta_1 \rangle - \mathbb{E}_n [\langle c', \theta_1 \rangle^2] + \mu' \|\alpha\|_1

which can be written as:

.. math::

    \min _{\rho \geq 0, \|\rho\|_1 \leq V_1} \max _{\omega_1 \geq 0, \|\omega_1\|_1 = 1} \ell(\rho, \omega_1)

where 

.. math::

    \ell(\rho, \omega_1) := 2 \omega_1^{\top} \mathbb{E}_n [u_1 y] - 2 \omega_1^{\top} \mathbb{E}_n [u_1 v_1^{\top}] \rho - \omega_1^{\top} \mathbb{E}_n [u_1 u_1^{\top}] \omega_1 + \mu' \sum_{i=1}^{2 p} \rho_i.

Moreover, :math:`v_1 = (a, -a)`, :math:`u_1 = (c', -c')`; and :math:`\theta_1 = \omega_1^{+} - \omega_1^{-}`, :math:`\alpha = \rho^+ - \rho^{-}`.


.. admonition:: FTRL iterates for Estimator 1
    :class: lemma
    :name: sparse-l1-l1-est1

    Consider the iterates for \(t=1,\ldots, T\):

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\rho}_{t+1} &= \exp\left(-\frac{\eta}{V_1} \left\{\sum_{\tau \leq t} -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1\tau} -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1t} + (t+1)\mu' \right\} - 1\right) \\
        \rho_{t+1} &=  \tilde{\rho}_{t+1} \min\left\{1, \frac{V_1}{\| \tilde{\rho}_{t+1} \|_1}\right\},
        \end{aligned}

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\omega}_{1,t+1} &= \tilde{\omega}_{1,t} \exp\bigg(2\eta\left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t}\right\} \\
        &\qquad -\eta\left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t-1} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t-1}\right\}\bigg) \\
        \omega_{1,t+1} &= \frac{\tilde{\omega}_{1,t+1}}{\|\tilde{\omega}_{1,t+1}\|_1}
        \end{aligned}

    with :math:`\tilde{\rho}_{-1} = \tilde{\rho}_{0} = \frac{1}{e}`, :math:`\tilde{\omega}_{1,-1} = \tilde{\omega}_{1,0} = \frac{1}{2p}`, and :math:`\eta = \frac{1}{8 \|\mathbb{E}_n [v_1 u_1^{\top}]\|_\infty}`.
    
    Then, :math:`\bar{\rho} = \frac{1}{T}\sum_{t=1}^{T} \rho_t`, :math:`\bar{\alpha} = \bar{\rho}^{+} - \bar{\rho}^{-}` is a :math:`O(T^{-1})`-approximate solution for :eq:`minimax-sparse-est1`.
    

**Proof**

The proof will match symbols with Proposition :ref:`proposition-17`. Let 

.. math::

    \Theta = \{\rho \;|\; \rho \geq 0,\, \|\rho\|_1 \leq V_1\}\;,\quad W = \{\omega_1 \;|\; \omega_1 \geq 0, \|\omega_1\|_1 = 1\}

be the convex feasibility sets. Note that :math:`\ell` is convex in :math:`\rho` and concave in :math:`\omega_1`. Since

.. math::

    \begin{aligned}
    \nabla_{\rho} \ell(\rho, \omega_1) &= -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_1 + \mu' \\
    \nabla_{\omega_1} \ell(\rho, \omega_1) &= 2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho - 2 \mathbb{E}_n [u_1 u_1^{\top}] \omega_1 
    \end{aligned}

the Lipschitzness property is satisfied with :math:`L = 2 \|\mathbb{E}_n [v_1 u_1^{\top}]\|_\infty`:

.. math::

    \begin{aligned}
    \left\|\nabla_\rho \ell(\rho, \omega_1) - \nabla_\rho \ell(\rho, \omega_1^{\prime})\right\|_{\infty} &= \left\|2 \mathbb{E}_n [v u^{\top}] (\omega_1 - \omega_1^{\prime})\right\|_{\infty} \leq 2 \|\mathbb{E}_n [v u^{\top}]\|_{\infty} \left\|\omega_1 - \omega_1^{\prime}\right\|_1 \\
    \left\|\nabla_{\omega_{1}} \ell(\rho, \omega_{1}) - \nabla_{\omega_{1}} \ell(\rho^{\prime}, \omega_{1})\right\|_{\infty} &= \left\|2 \mathbb{E}_n [u v^{\top}] (\rho - \rho^{\prime})\right\|_{\infty} \leq 2 \|\mathbb{E}_n [v u^{\top}]\|_{\infty} \left\|\rho - \rho^{\prime}\right\|_1
    \end{aligned}

Consider the entropic regularizers :math:`R_{min}(\rho) = V_1 \sum_{i=1}^{2p} \rho_i \log (\rho_i)`, and :math:`R_{max}(\omega_1) = \sum_{i=1}^{2p} \omega_{1i} \log (\omega_{1i})` which are :math:`1`-strongly convex in the spaces :math:`\Theta`, and :math:`W` respectively. Then, the iterates satisfy:

.. math::
    :nowrap:

    \begin{aligned}
    \rho_{t+1} &= \underset{\rho \geq 0, \|\rho\|_1 \leq V_1}{\operatorname{argmin}} \rho^{\top} \left(\sum_{\tau \leq t} \left\{-2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1\tau} + \mu'\right\} - 2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1t} + \mu'\right) + \frac{V_1}{\eta} \sum_{i=1}^{2p} \rho_i \log (\rho_i) \\
    \tilde{\rho}_{t+1} &= \exp\left(-\frac{\eta}{V_1} \left\{\sum_{\tau \leq t} -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1\tau} -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1t} + (t+1)\mu' \right\} - 1\right) \\
    \rho_{t+1} &=  \tilde{\rho}_{t+1} \min\left\{1, \frac{V_1}{\| \tilde{\rho}_{t+1} \|_1}\right\},
    \end{aligned}

.. math::
    :nowrap:

    \begin{aligned}
    \omega_{1,t+1} &= \underset{\|\omega_1\|_1 \leq 1}{\operatorname{argmax}} \omega_1^{\top} \left(\sum_{\tau \leq t} \left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{\tau} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \omega_{1\tau} \right\} \\
    &\qquad + 2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \omega_{1t} \right) - \frac{1}{\eta} \sum_{i=1}^{2p} \omega_{1i} \log (\omega_{1i}) \\
    \tilde{\omega}_{1,t+1} &= \tilde{\omega}_{1,t} \exp\left(2\eta \left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t}\right\} \\
    &\qquad -\eta \left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t-1} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t-1}\right\}\right) \\
    \omega_{1,t+1} &= \frac{\tilde{\omega}_{1,t+1}}{\|\tilde{\omega}_{1,t+1}\|_1}
    \end{aligned}

with :math:`\omega_{1,-1} = \omega_{1,0} = \frac{1}{2p}`. Therefore, by Proposition :ref:`proposition-17`, the ensemble

.. math::

    \bar{\rho} = \frac{1}{T} \sum_{t=1}^T \rho_t

is :math:`O\left(\frac{1}{T}\right)`-approximate solution for the minimax objective.

.. admonition:: Duality Gap
    :class: note

   The ensembles :math:`\bar{\alpha}`, :math:`\bar{\theta_1}` can be thought of as primal and dual solutions and we can use the duality gap as a certificate for convergence of the algorithm.

.. math::
    :nowrap:

    \begin{aligned}
    \text { Duality Gap } &:= \max _{\|\theta_1\|_1 \leq 1 } L(\bar{\alpha}, \theta_1) - \min _{\|\alpha\|_1 \leq V_1} L(\alpha, \bar{\theta_1}) \\
    &\leq \left(\mathbb{E}_n [(y - \langle \bar{\alpha}, a \rangle)c']\right)^{\top} \mathbb{E}_n [c' c'^{\top}]^{\dagger} \left(\mathbb{E}_n [(y - \langle \bar{\alpha}, a \rangle)c']\right) + \mu' \|\bar{\alpha}\|_1 \\
    &\quad - \left(\bar{\theta_1}^{\top} \mathbb{E}_n [c'y] + V_1 \left\{\mu' - 2 \|\mathbb{E}_n [a c'^{\top}] \bar{\theta_1}\|_\infty \right\}^{-} - \bar{\theta_1}^{\top} \mathbb{E}_n [c' c'^{\top}] \bar{\theta_1}\right) := \text{ tol}
    \end{aligned}

.. _estimator-2:

Estimator 2
===========

The ridge estimator takes the form:

.. math::
    :label: minimax-sparse-est2

    \hat{\alpha} := \argmin_{\|\alpha\|_1 \leq V_1} \max _{\|\theta_1\|_1 \leq 1} 2 \langle \mathbb{E}_n [(y - \langle \alpha, a \rangle)c'], \theta_1 \rangle - \mathbb{E}_n [\langle c', \theta_1 \rangle^2] + \mu' \mathbb{E}_n [\langle a, \alpha \rangle^2]

This estimator can be shown to solve the problem:

.. math::

    \min _{\rho \geq 0, \|\rho\|_1 \leq V_1} \max _{\omega_1 \geq 0, \|\omega_1\|_1 = 1} \ell(\rho, \omega_1)

where 

.. math::

    \ell(\rho, \omega_1) := 2 \omega_1^{\top} \mathbb{E}_n [u_1 y] - 2 \omega_1^{\top} \mathbb{E}_n [u_1 v_1^{\top}] \rho - \omega_1^{\top} \mathbb{E}_n [u_1 u_1^{\top}] \omega_1 + \mu' \rho^{\top} \mathbb{E}_n [v_1 v_1^{\top}] \rho

Moreover, :math:`v_1 = (a, -a)`, :math:`u_1 = (c', -c')`; and :math:`\theta_1 = \omega_1^{+} - \omega_1^{-}`, :math:`\alpha = \rho^+ - \rho^{-}`.

.. admonition:: FTRL iterates for Estimator 2
    :class: lemma
    :name: sparse-l1-l1-est2

    Consider the iterates for :math:`t = 1, \ldots, T`:

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\rho}_{t+1} &= \exp\left(-\frac{\eta}{V_1} \left\{\sum_{\tau \leq t} -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1\tau} + 2 \mu' \mathbb{E}_n [v_1 v_1^{\top}] \tilde{\rho}_{\tau} - 2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1t} + 2 \mu' \mathbb{E}_n [v_1 v_1^{\top}] \tilde{\rho}_{t} \right\} - 1\right) \\
        \rho_{t+1} &= \tilde{\rho}_{t+1} \min\left\{1, \frac{V_1}{\| \tilde{\rho}_{t+1} \|_1}\right\},
        \end{aligned}

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\omega}_{1,t+1} &= \tilde{\omega}_{1,t} \exp\bigg(2\eta\left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t}\right\} \\
        &\qquad -\eta\left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t-1} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t-1}\right\}\bigg) \\
        \omega_{1,t+1} &= \frac{\tilde{\omega}_{1,t+1}}{\|\tilde{\omega}_{1,t+1}\|_1}
        \end{aligned}

    with :math:`\tilde{\rho}_{-1} = \tilde{\rho}_{0} = \frac{1}{e}`, :math:`\tilde{\omega}_{1,-1} = \tilde{\omega}_{1,0} = \frac{1}{2p}`, and :math:`\eta = \frac{1}{8 \|\mathbb{E}_n [v_1 u_1^{\top}]\|_\infty}`.

    Then, :math:`\bar{\rho} = \frac{1}{T} \sum_{t=1}^{T} \rho_t`, :math:`\bar{\alpha} = \bar{\rho}^{+} - \bar{\rho}^{-}` is a :math:`O(T^{-1})`-approximate solution for :eq:`minimax-sparse-est2`.

**Proof**

The proof is analogous to :ref:`sparse-l1-l1-est1`.

.. admonition:: Duality gap
    :class: remark

    The upper bound for the duality gap as a certificate for convergence of the algorithm is given by:

    .. math::
        :nowrap:

        \begin{aligned}
        \text { tol } &= \left(\mathbb{E}_n [(y - \langle \bar{\alpha}, a \rangle)c']\right)^{\top} \mathbb{E}_n [c' c'^{\top}]^{\dagger} \left(\mathbb{E}_n [(y - \langle \bar{\alpha}, a \rangle)c']\right) + \mu' \bar{\alpha}^{\top} \mathbb{E}_n [aa^{\top}] \bar{\alpha} \\
        &\quad - \left(2 \bar{\theta_1}^{\top} \mathbb{E}_n [c'y] - \bar{\theta_1}^{\top} \mathbb{E}_n [c'a^{\top}] \frac{\mathbb{E}_n [aa^{\top}]^{\dagger}}{\mu'} \mathbb{E}_n [ac'^{\top}] \bar{\theta_1} - \bar{\theta_1}^{\top} \mathbb{E}_n [c' c'^{\top}] \bar{\theta_1} \right)
        \end{aligned}
