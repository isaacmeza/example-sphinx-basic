.. _sparse-linear-function-spaces:

Sparse Linear Function Spaces ($\ell_1-\ell_1$)
===============================================

In this section we address the high-dimensional case, where the function class is sparse linear, i.e. :math:`g(a) = \langle \alpha, a\rangle`, where :math:`\|\alpha\|_0 := \{j\in [p]\,|\,|\alpha_j|>0\} \leq s`. We will consider :math:`\ell_1` relaxations for the minimax optimization problem with :math:`\ell_1`-balls for the adversary. We remove the non-smoothness of the :math:`\ell_1` regularization by lifting the parameter :math:`\alpha` to a :math:`2p`-dimensional positive orthant. Consider two vectors :math:`\rho^{+}, \rho^{-} \geq 0` and then setting :math:`\alpha = \rho^{+} - \rho^{-}`, with :math:`\rho = \left(\rho^{+}; \rho^{-}\right)`. Observe that for any feasible :math:`\bar{\alpha}`, the solution :math:`\rho_i^{+} = \alpha_i 1\left\{\alpha_i > 0\right\}` and :math:`\rho_i^{-} = \alpha_i 1\left\{\alpha_i \leq 0\right\}` is still feasible and achieves the same objective, by the linearity of the loss function. Moreover, any solution :math:`\rho`, maps to a feasible solution :math:`\alpha` and thus the two optimization programs have the same optimal solutions.

Thus we will be solving an optimization problem over the :math:`2p`-dimensional simplex, and we will be using *Optimistic-Follow-the-Regularized-Leader* to find an :math:`\epsilon`-approximate solution. The approximate solutions of the minimax problems for all of our estimator will rely on the following proposition:

.. blockquote:: Proposition 17 in `Dikkala et al. (2020) <https://arxiv.org/abs/2006.07201>`_
   :class: lemma

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
