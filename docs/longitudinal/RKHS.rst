Regularized Kernel Hilbert Space
================================
.. _rkhs_estimators:

In this section we assume that the function classes 
whenever :math:`\mathcal{G}`, :math:`\mathcal{H}`, :math:`\mathcal{F}`, :math:`\mathcal{F}^\prime` are RKHS.  Let :math:`\Phi_A:\mathcal{G}\rightarrow\mathbb{R}^n` be an operator with :math:`i`th row :math:`\langle \phi(A_i), \cdot \rangle_{\mathcal{G}}` with corresponding kernel matrix :math:`K_A`.  Define analogously :math:`\Phi_B, \ldots` for the rest of the function classes.


Closed form - Estimator 1
-------------------------

We study the estimator

.. math::

    \hat{g} = \arg \min_{g \in \mathcal{G}} 
    \max_{f' \in \mathcal{F'}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right] - \lambda \| f \|_{\mathcal{F}}^2
     + \mu' \| g \|_{\mathcal{G}}^2

.. admonition:: Formula of minimizers

    The minimizer takes the form :math:`\hat{g} = \Phi_A^* \hat{\alpha}` where,

    .. math::

        \hat{\alpha} &= \left(K_A P_C' K_A + \mu K_A \right)^{\dagger} K_A P_C' Y \\
        P_{C'} &= \left(K_{C'} + \lambda \right)^{\dagger} K_{C'}


Closed form - Estimator 2
-------------------------

We study the estimator

.. math::

    \hat{g} = \arg \min_{g \in \mathcal{G}} 
    \max_{f' \in \mathcal{F'}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right]
     + \mu' \mathbb{E}_n \{ g(A)^2 \}

.. admonition:: Formula of minimizers

    The minimizer takes the form :math:`\hat{g} = \Phi_A^* \hat{\alpha}` where,

    .. math::

        \hat{\alpha} &= \left( K_A P_C' K_A + \mu K_A^2 \right)^{\dagger} K_A P_C' Y \\
        P_{C'} &= K_{C'}^{\dagger} K_{C'}

We study the ridge regularized *joint* estimator:

.. math::

    (\hat{g}, \hat{h}) = \arg \min_{g \in \mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right]
    + \mu' \mathbb{E}_n \{ g(A)^2 \} \\
    \quad + \max_{f \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ h(B) - g(A) \right\} f(C) - f(C)^2 \right]
    + \mu \mathbb{E}_n \{ h(B)^2 \}

Let :math:`V_{g,h}' = g(A) - Y` and :math:`V_{g,h} = h(B) - g(A)`. Let :math:`\Phi_C : \mathcal{F} \rightarrow \mathbb{R}^n` be an operator with :math:`i`th row :math:`\langle \phi(C_i), \cdot \rangle_{\mathcal{F}}`. Define :math:`\Phi_{C'}` analogously, replacing :math:`C_i` with :math:`C_i'`. Let :math:`K_C` and :math:`K_{C'}` be the corresponding kernel matrices.

In remarks below, we also study the following modification, which we call the "subsetted" estimator:

.. math::

    (\hat{g}, \hat{h}) = \arg \min_{g \in \mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_p \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right]
    + \mu' \mathbb{E}_n \{ g(A)^2 \} \\
    \quad + \max_{f \in \mathcal{F}} \mathbb{E}_q \left[ 2 \left\{ h(B) - g(A) \right\} f(C) - f(C)^2 \right]
    + \mu \mathbb{E}_n \{ h(B)^2 \}

where :math:`[p]` and :math:`[q]` partition :math:`[n] = (1, \ldots, n)`, so :math:`p + q = n`.

For the index set :math:`[p]`, let :math:`I_{[p]} \in \mathbb{R}^{p \times n}` be the matrix of ones and zeros such that :math:`V_{[p]} = I_{[p]} V` gives the elements of :math:`V` whose indices are in :math:`[p]`.

Maximizers
----------

**Existence of maximizers**

There exist coefficients :math:`\hat{\gamma}_{g,h}, \hat{\gamma}'_{g,h} \in \mathbb{R}^n` such that maximizers take the form :math:`\hat{f}_{g,h} = \Phi_C^* \hat{\gamma}_{g,h}` and :math:`\hat{f}'_{g,h} = \Phi_{C'}^* \hat{\gamma}'_{g,h}`.

**Remark (Subsetted estimator)**

For the subsetted estimator, the same results hold but with :math:`\hat{\gamma}_{g,h;[q]} \in \mathbb{R}^q` and :math:`\hat{\gamma}'_{g,h;[p]} \in \mathbb{R}^p`, acting on appropriately modified feature operators :math:`\Phi^*_{C;[q]}` and :math:`\Phi^*_{C';[p]}`.

**Proof**

Write the objectives for the maximizers as

.. math::

    \mathcal{E}'(f') = \mathbb{E}_n \left\{ 2 V'_{g,h} f'(C') - f'(C')^2 \right\} \\
    \mathcal{E}(f) = \mathbb{E}_n \left\{ 2 V_{g,h} f(C) - f(C)^2 \right\}

We prove the former result; the latter is similar. By the Riesz representation theorem,

.. math::

    \mathcal{E}(f) = \mathbb{E}_n \left\{ 2 V_{g,h} \langle f, \phi(C) \rangle_{\mathcal{F}} - \langle f, \phi(C) \rangle_{\mathcal{F}}^2 \right\}

For an RKHS, evaluation is a continuous functional represented as the inner product with the feature map. Due to the ridge penalty, the stated objective has a maximizer :math:`\hat{f}_{g,h}` that obtains the maximum.

To lighten notation, we suppress the indexing of :math:`\hat{f}_{g,h}` by :math:`(g,h)` for the rest of this argument. Write :math:`\hat{f} = \hat{f}_n + \hat{f}^{\perp}_n` where :math:`\hat{f}_n \in \text{row}(\Phi_C)` and :math:`\hat{f}_n^{\perp} \in \text{null}(\Phi_C)`. Substituting this decomposition of :math:`\hat{f}` into the objective, we see that

.. math::

    \mathcal{E}(\hat{f}) = \mathcal{E}(\hat{f}_n)

Hence if :math:`\hat{f}` is a maximizer, then there exists :math:`\hat{f}_n` that is also a maximizer.

**Formula of maximizers**

The explicit formula for the coefficients is :math:`\hat{\gamma}_{g,h} = K_C^{\dagger} \vec{V}_{g,h}` and :math:`\hat{\gamma}'_{g,h} = K_{C'}^{\dagger} \vec{V}'_{g,h}`.

**Remark (Subsetted estimator)**

For the subsetted estimator, the same results hold but with :math:`\hat{\gamma}_{g,h;[q]} = K_{C;[q,q]}^{\dagger} \vec{V}_{g,h;[q]}` and :math:`\hat{\gamma}'_{g,h;[p]} = K_{C';[p,p]}^{\dagger} \vec{V}'_{g,h;[p]}`.

**Proof**

We prove the former result; the latter is similar. Write the objective as

.. math::

    \mathcal{E}(f) = 2 \langle f, \hat{\mu}_{g,h} \rangle_{\mathcal{F}} - \langle f, \hat{T}_C f \rangle_{\mathcal{F}}

where :math:`\hat{\mu}_{g,h} = \mathbb{E}_n \{ V_{g,h} \phi(C) \} = \frac{1}{n} \Phi_C^* \vec{V}_{g,h}` and :math:`\hat{T}_C = \mathbb{E}_n \{ \phi(C) \otimes \phi(C)^* \} = \frac{1}{n} \Phi_C^* \Phi_C`. Hence by the existence of maximizers,

.. math::

    \mathcal{E}(\gamma) = 2 \langle \Phi_C^* \gamma_{g,h}, \hat{\mu}_{g,h} \rangle_{\mathcal{F}} - \langle \Phi_C^* \gamma_{g,h}, \hat{T}_C \Phi_C^* \gamma_{g,h} \rangle_{\mathcal{F}}
    = \frac{2}{n} \gamma_{g,h}^{\top} \Phi_C \Phi_C^* \vec{V}_{g,h} - \frac{1}{n} \gamma_{g_h}^{\top} \Phi_C \Phi_C^* \Phi_C \Phi_C^* \gamma_{g,h}

Since :math:`K_C = \Phi_C \Phi_C^*`, the first order condition yields :math:`K_C \vec{V}_{g,h} = K_C^2 \hat{\gamma}_{g,h}`, i.e. :math:`\hat{\gamma}_{g,h} = K_C^{\dagger} \vec{V}_{g,h}` where :math:`K_C^{\dagger}` is the pseudoinverse of :math:`K_C`.

Minimizers
----------

Let :math:`\Phi_A : \mathcal{H} \rightarrow \mathbb{R}^n` be an operator with :math:`i`th row :math:`\langle \phi(A_i), \cdot \rangle_{\mathcal{H}}`. Define :math:`\Phi_B` analogously, replacing :math:`A_i` with :math:`B_i`. Let :math:`K_A` and :math:`K_B` be the corresponding kernel matrices.

**Existence of minimizers**

There exist coefficients :math:`\alpha, \beta \in \mathbb{R}^n` such that minimizers take the form :math:`\hat{g} = \Phi_A^* \hat{\alpha}` and :math:`\hat{h} = \Phi_B^* \hat{\beta}`.

**Remark (Subsetted estimator)**

The result remains true for the subsetted estimator.

**Proof**

To begin, write the objective :math:`\mathcal{E}(g,h)` as

.. math::

    \mathbb{E}_n \left\{ 2 V'_{g,h} \hat{f}_{g,f}'(C') - \hat{f}_{g,h}'(C')^2 \right\}
    + \mu' \mathbb{E}_n \{ g(A)^2 \} \\
    + \mathbb{E}_n \left\{ 2 V_{g,h} \hat{f}_{g,h}(C) - \hat{f}_{g,h}(C)^2 \right\}
    + \mu \mathbb{E}_n \{ h(B)^2 \}

By the existence and formula of maximizers,

.. math::

    \hat{f}_{g,f}'(C') = \langle \hat{f}_{g,f}', \phi(C') \rangle_{\mathcal{F}}
    = \langle \Phi_{C'}^* K_{C'}^{\dagger} \vec{V}'_{g,h}, \phi(C') \rangle_{\mathcal{F}} \\
    \hat{f}_{g,h}(C) = \langle \hat{f}_{g,f}, \phi(C) \rangle_{\mathcal{F}}
    = \langle \Phi_{C}^* K_{C}^{\dagger} \vec{V}_{g,h}, \phi(C) \rangle_{\mathcal{F}}

Hence :math:`(g,h)` only appear via :math:`V'_{g,h} = g(A) - Y`, :math:`V_{g,h} = h(B) - g(A)`, and directly as :math:`g(A)` and :math:`h(B)`. In all of these expressions, they can be further expressed as :math:`g(A) = \langle g, \phi(A) \rangle_{\mathcal{G}}` and :math:`h(B) = \langle h, \phi(B) \rangle_{\mathcal{H}}`, which is a linear functional. The overall objective is quadratic in such terms, so the stated objective has maximizers :math:`(\hat{g}, \hat{h})` that obtain the maximum.

By a similar argument to the existence of maximizers, for any :math:`(\hat{g}, \hat{h})` attaining the maximum, :math:`\mathcal{E}(\hat{g}, \hat{h}) = \mathcal{E}(\hat{g}_n, \hat{h}_n)` where :math:`\hat{g}_n \in \text{row}(\Phi_A)` and :math:`\hat{h}_n \in \text{row}(\Phi_B)`.

**Properties of pseudo-inverse**

For any square symmetric matrix :math:`K \in \mathbb{R}^{n \times n}`, its eigendecomposition is :math:`K = U \Sigma U^{\top}` where :math:`\Sigma \in \mathbb{R}^{r \times r}` and :math:`r \leq n`. Its pseudo-inverse is :math:`K^- = U \Sigma^{\dagger} U^{\top}`.
