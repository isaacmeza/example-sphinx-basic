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

    \hat{g}=\arg \min_{g\in\mathcal{G}} 
    \max_{f' \in \mathcal{F'}} \mathbb{E}_n\left[2\left\{g(A)-Y\right\} f'(C')-f'(C')^2\right]
     +\mu'\E_n\{g(A)^2\} 

.. lemma:: Formula of minimizers
   :name: lemma:min_2

    The minimizer takes the form $\hat{g} = \Phi_A^*\hat\alpha$ where,

    .. math::

        \hat{\alpha} &= \left(K_A P_C' K_A + \mu K_A^2\right)^{\dagger}K_AP_C'Y\\
        P_{C'}&=K_{C'}^{\dagger}K_{C'}

We study the ridge regularized *joint* estimator:

.. math::

    (\hat{g},\hat{h})=\arg \min _{g\in\mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_n\left[2\left\{g(A)-Y\right\} f'(C')-f'(C')^2\right]
     +\mu'\E_n\{g(A)^2\} \\
    &\quad +
    \max_{f \in \mathcal{F}} \mathbb{E}_n\left[2\left\{h(B)-g(A)\right\} f(C)-f(C)^2\right]   
    +\mu\E_n\{h(B)^2\}

Let $V_{g,h}'=g(A)-Y$ and $V_{g,h}=h(B)-g(A)$. Let $\Phi_C:\mathcal{F}\rightarrow\mathbb{R}^n$ be an operator with $i$th row $\langle \phi(C_i),\cdot \rangle_{\mathcal{F}}$. Define $\Phi_{C'}$ analogously, replacing $C_i$ with $C_i'$. Let $K_C$ and $K_{C'}$ be the corresponding kernel matrices.

In remarks below, we also study the following modification, which we call the "subsetted" estimator:

.. math::

    (\hat{g},\hat{h})=\arg \min _{g\in\mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_p\left[2\left\{g(A)-Y\right\} f'(C')-f'(C')^2\right]
     +\mu'\E_n\{g(A)^2\} \\
    &\quad +
    \max_{f \in \mathcal{F}} \mathbb{E}_q\left[2\left\{h(B)-g(A)\right\} f(C)-f(C)^2\right]   
    +\mu\E_n\{h(B)^2\}

where $[p]$ and $[q]$ partition $[n]=(1,...,n)$, so $p+q=n$. 

For the index set $[p]$, let $I_{[p]}\in\mathbb{R}^{p\times n}$ be the matrix of ones and zeros such that $V_{[p]}=I_{[p]}V$ gives the elements of $V$ whose indices are in $[p]$.

Maximizers
----------

.. lemma:: Existence of maximizers
   :name: lemma:max_exist

    There exist coefficients $\hat{\gamma}_{g,h},\hat{\gamma}'_{g,h}\in\mathbb{R}^n$ such that maximizers take the form $\hat{f}_{g,h}=\Phi_C^* \hat{\gamma}_{g,h}$ and $\hat{f}'_{g,h}=\Phi_{C'}^*\hat{\gamma}'_{g,h}$.

.. remark:: Subsetted estimator
   :name: remark:max_exist

    For the subsetted estimator, the same results hold but with $\hat{\gamma}_{g,h;[q]}\in\mathbb{R}^q$ and $\hat{\gamma}'_{g,h;[p]}\in\mathbb{R}^p$, acting on appropriately modified feature operators $\Phi^*_{C;[q]}$ and $\Phi^*_{C';[p]}$.

.. proof::

    Write the objectives for the maximizers as

    .. math::

        \mathcal{E}'(f')&=\mathbb{E}_n\left\{2V'_{g,h} f'(C')-f'(C')^2\right\} \\
        \mathcal{E}(f)&=\mathbb{E}_n\left\{2V_{g,h} f(C)-f(C)^2\right\}.

    We prove the former result; the latter is similar. By the Riesz representation theorem,

    .. math::

        \mathcal{E}(f)=\mathbb{E}_n\left\{2V_{g,h} \langle f, \phi(C)\rangle_{\mathcal{F}}-\langle f, \phi(C)\rangle_{\mathcal{F}}^2\right\}.

    For an RKHS, evaluation is a continuous functional represented as the inner product with the feature map. Due to the ridge penalty, the stated objective has a maximizer $\hat{f}_{g,h}$ that obtains the maximum.

    To lighten notation, we suppress the indexing of $\hat{f}_{g,h}$ by $(g,h)$ for the rest of this argument. Write $\hat{f}=\hat{f}_n+\hat{f}^{\perp}_n$ where $\hat{f}_n\in row(\Phi_C)$ and $\hat{f}_n^{\perp}\in null(\Phi_C)$. Substituting this decomposition of $\hat{f}$ into the objective, we see that

    .. math::

        \mathcal{E}(\hat{f})=\mathcal{E}(\hat{f}_n).

    Hence if $\hat{f}$ is a maximizer, then there exists $\hat{f}_n$ that is also a maximizer.

.. lemma:: Formula of maximizers
   :name: lemma:max

    The explicit formula for the coefficients is $\hat{\gamma}_{g,h}=K_C^{\dagger}\vec{V}_{g,h}$ and $\hat{\gamma}'_{g,h}=K_{C'}^{\dagger}\vec{V}'_{g,h}$.

.. remark:: Subsetted estimator
   :name: remark:max

    For the subsetted estimator, the same results hold but with $\hat{\gamma}_{g,h;[q]}=K_{C;[q,q]}^{\dagger}\vec{V}_{g,h;[q]}$ and $\hat{\gamma}'_{g,h;[p]}=K_{C';[p,p]}^{\dagger}\vec{V}'_{g,h;[p]}$.

.. proof::

    We prove the former result; the latter is similar. Write the objective as

    .. math::

        \mathcal{E}(f)= 2\langle f, \hat{\mu}_{g,h}\rangle_{\mathcal{F}}-\langle f, \hat{T}_C f\rangle_{\mathcal{F}}.

    where $\hat{\mu}_{g,h}=\mathbb{E}_n\{V_{g,h}\phi(C)\}=\frac{1}{n}\Phi_C^* \vec{V}_{g,h}$ and $\hat{T}_C=\mathbb{E}_n\{\phi(C)\otimes \phi(C)^*\}=\frac{1}{n}\Phi_C^*\Phi_C$. Hence by :ref:`lemma:max_exist`,

    .. math::

        \mathcal{E}(\gamma)= 2\langle \Phi_C^* \gamma_{g,h}, \hat{\mu}_{g,h}\rangle_{\mathcal{F}}-\langle \Phi_C^* \gamma_{g,h}, \hat{T}_C \Phi_C^* \gamma_{g,h}\rangle_{\mathcal{F}}=\frac{2}{n}\gamma_{g,h}^{\top}\Phi_C \Phi_C^* \vec{V}_{g,h}-\frac{1}{n}\gamma_{g,h}^{\top} \Phi_C \Phi_C^*\Phi_C \Phi_C^* \gamma_{g,h}.

    Since $K_C=\Phi_C\Phi_C^*$, the first order condition yields $K_C\vec{V}_{g,h}=K_C^2 \hat{\gamma}_{g,h}$, i.e. $\hat{\gamma}_{g,h}=K_C^{\dagger}\vec{V}_{g,h}$ where $K_C^{\dagger}$ is the pseudoinverse of $K_C$.

Minimizers
----------

Let $\Phi_A:\mathcal{H}\rightarrow\mathbb{R}^n$ be an operator with $i$th row $\langle \phi(A_i),\cdot \rangle_{\mathcal{H}}$. Define $\Phi_B$ analogously, replacing $A_i$ with $B_i$. Let $K_A$ and $K_B$ be the corresponding kernel matrices.

.. lemma:: Existence of minimizers
   :name: lemma:min_exist

    There exist coefficients $\alpha,\beta \in\mathbb{R}^n$ such that minimizers take the form $\hat{g}=\Phi_A^*\hat{\alpha}$ and $\hat{h}=\Phi_B^*\hat{\beta}$.

.. remark:: Subsetted estimator
   :name: remark:min_exist

    The result remains true for the subsetted estimator.

.. proof::

    To begin, write the objective $\mathcal{E}(g,h)$ as 

    .. math::

       \mathbb{E}_n\left\{2V'_{g,h} \hat{f}_{g,f}'(C')-\hat{f}_{g,h}'(C')^2\right\}
         +\mu'\E_n\{g(A)^2\} 
        +
         \mathbb{E}_n\left\{2V_{g,h} \hat{f}_{g,h}(C)-\hat{f}_{g,h}(C)^2\right\}   
        +\mu\E_n\{h(B)^2\}.

     By :ref:`lemma:max_exist` and :ref:`lemma:max`,

    .. math::

         \hat{f}_{g,f}'(C') =\langle \hat{f}_{g,f}',  \phi(C')\rangle_{\mathcal{F}} =\langle \Phi_{C'}^*K_{C'}^{\dagger}\vec{V}'_{g,h},  \phi(C')\rangle_{\mathcal{F}} \\
         \hat{f}_{g,h}(C) =\langle \hat{f}_{g,f},  \phi(C)\rangle_{\mathcal{F}} =\langle \Phi_{C}^*K_{C}^{\dagger}\vec{V}_{g,h},  \phi(C)\rangle_{\mathcal{F}}.

     Hence $(g,h)$ only appear via $V'_{g,h}=g(A)-Y$, $V_{g,h}=h(B)-g(A)$, and directly as $g(A)$ and $h(B)$. In all of these expressions, they can be further expressed as $g(A)=\langle g,\phi(A)\rangle_{\mathcal{G}}$ and $h(B)=\langle h,\phi(B)\rangle_{\mathcal{H}}$, which is a linear functional. The overall objective is quadratic in such terms, so the stated objective has minimizers $(\hat{g},\hat{h})$ that obtain the minimum.

     By a similar argument to :ref:`lemma:max_exist`, for any $(\hat{g},\hat{h})$ attaining the minimum, $\mathcal{E}(\hat{g},\hat{h})=\mathcal{E}(\hat{g}_n,\hat{h}_n)$ where $\hat{g}_n\in row(\Phi_A)$ and $\hat{h}_n\in row(\Phi_B)$.

.. lemma:: Formula of minimizers
   :name: lemma:min

    The explicit formula for the coefficients is 

    .. math::

        \hat{\beta} &= \left[K_A\left\{P_C+\left(P_{C'}+P_C+\mu'\right)K_A\left(K_BP_CK_A\right)^{\dagger}K_B\left(P_C+\mu\right)\right\}K_B\right]^{\dagger}K_AP_{C'}Y\\
        \hat{\alpha}&=  \left(K_BP_CK_A\right)^{\dagger}K_B\left(P_C+\mu\right)K_B\hat{\beta}      

.. proof::

    We proceed in steps.

    1. Write the objective $\mathcal{E}(g,h)$ as

    .. math::

       2\langle \hat{f}'_{g,h}, \hat{\mu}'_{g,h}\rangle_{\mathcal{F}}-\langle \hat{f}'_{g,h}, \hat{T}_{C'} \hat{f}'_{g,h}\rangle_{\mathcal{F}}  
         +\mu'\langle g,\hat{T}_A g\rangle_{\mathcal{G}} 
        +
        2\langle \hat{f}_{g,h}, \hat{\mu}_{g,h}\rangle_{\mathcal{F}}-\langle \hat{f}_{g,h}, \hat{T}_C \hat{f}_{g,h}\rangle_{\mathcal{F}}  
        +\mu\langle h,\hat{T}_B h\rangle_{\mathcal{H}}.

    where 
    $\hat{\mu}'_{g,h}=\frac{1}{n}\Phi_{C'}^* \vec{V}'_{g,h}$, 
    $\hat{\mu}_{g,h}=\frac{1}{n}\Phi_C^* \vec{V}_{g,h}$, and the covariance operators are defined analogously to :ref:`lemma:max`. Hence by :ref:`lemma:max`,

    .. math::

        \mathcal{E}(g,h)
        &=\frac{2}{n} (\vec{V}'_{g,h})^{\top}K_{C'}^{\dagger}\Phi_{C'}\Phi_{C'}^* \vec{V}'_{g,h}
        -\frac{1}{n}(\vec{V}'_{g,h})^{\top}K_{C'}^{\dagger}\Phi_{C'} \Phi_{C'}^*\Phi_{C'}  \Phi_{C'}^*K_{C'}^{\dagger}\vec{V}'_{g,h} 
         +\mu'\langle g,\hat{T}_A g\rangle_{\mathcal{G}}  \\
        &+\frac{2}{n}\vec{V}_{g,h}^{\top}K_{C}^{\dagger}\Phi_{C} \Phi_C^* \vec{V}_{g,h}
        -\frac{1}{n}\vec{V}_{g,h}^{\top}K_{C}^{\dagger}\Phi_{C} \Phi_{C}^*\Phi_{C} \Phi_{C}^*K_{C}^{\dagger}\vec{V}_{g,h}  
        +\mu\langle h,\hat{T}_B h\rangle_{\mathcal{H}} \\
        &=\frac{1}{n}(\vec{V}'_{g,h})^{\top} P_{C'}\vec{V}'_{g,h}
         +\mu'\langle g,\hat{T}_A g\rangle_{\mathcal{G}}  +
        \frac{1}{n}\vec{V_{g,h}}^{\top}P_C\vec{V}_{g,h}
        +\mu\langle h,\hat{T}_B h\rangle_{\mathcal{H}}.

    2. Let $Y,G,H\in\mathbb{R}^n$ be defined with $G_i=g(A_i)$ and $H_i=h(B_i)$. In this notation,

    .. math::

        \frac{1}{n}(\vec{V}'_{g,h})^{\top} P_{C'}\vec{V}'_{g,h} 
        &=\frac{1}{n}(Y^{\top}P_{C'}Y-2G^{\top}(P_{C'}Y+P_CH)+G^{\top}(P_{C'}+P_C+\mu')G+H^{\top}(P_C+\mu)H).

    Combining with $G=\Phi_Ag=K_A\alpha$ and $H=\Phi_B h=K_B\beta$ from :ref:`lemma:min_exist`,

    .. math::

        n\mathcal{E}(\alpha,\beta)&=Y^{\top}P_{C'}Y-2G^{\top}(P_{C'}Y+P_CH)+\alpha^{\top}K_A(P_{C'}+P_C+\mu') K_A\alpha\\
        &\quad +\beta^{\top}K_B (P_C+\mu) K_B\beta.

    3. The first order conditions yield

    .. math::

        0&=-2K_A(P_{C'}Y+P_CK_B\hat{\beta})+2 K_A(P_{C'}+P_C+\mu') K_A\hat{\alpha} \\
        0&=-2K_BP_C K_A\hat{\alpha}+2K_B (P_C+\mu) K_B \hat{\beta} \Longrightarrow \hat{\alpha} = \left(K_BP_CK_A\right)^{\dagger}K_B\left(P_C+\mu\right)K_B\hat{\beta}.

    4. Substituting the latter into the former,

    .. math::

        K_AP_{C'}Y+K_AP_CK_B\hat{\beta}=K_A(P_{C'}+P_C+\mu') K_A\left(K_BP_CK_A\right)^{\dagger}K_B\left(P_C+\mu\right)K_B\hat{\beta},

    and solving for $\hat{\beta}$,

    .. math::

        \hat{\beta} = \left[K_A\left\{P_C+\left(P_{C'}+P_C+\mu'\right)K_A\left(K_BP_CK_A\right)^{\dagger}K_B\left(P_C+\mu\right)\right\}K_B\right]^{\dagger}K_AP_{C'}Y.

.. remark:: Subsetted estimator
   :name: remark:min

    The explicit formula for the coefficients is 

    .. math::

        \hat{\beta} &= \left[K_A\left\{\tilde{P}_C+\left(\tilde{P}_{C'}+\tilde{P}_C+\mu'\right)K_A\left(K_B\tilde{P}_CK_A\right)^{\dagger}K_B\left(\tilde{P}_C+\mu\right)\right\}K_B\right]^{\dagger}K_A\tilde{P}_{C'}Y\\
        \hat{\alpha}&=  \left(K_B\tilde{P}_CK_A\right)^{\dagger}K_B\left(\tilde{P}_C+\mu\right)K_B\hat{\beta}      

    where $\tilde{P}_{C'}=\frac{n}{p}I_{[p]}^{\top}P_{C';[p,p]}I_{[p]}$ and $\tilde{P}_{C}=\frac{n}{q}I_{[q]}^{\top}P_{C;[q,q]}I_{[q]}$. Note that $P_{C';[p,p]}=(K_{C';[p,p]})^-K_{C';[p,p]}$ and  $K_{C';[p,p]}=I_{[p]}K_{C'}I_{[p]}^{\top}$.

.. proof::

    We proceed in steps.

    1. Write the objective $\mathcal{E}(g,h)$ as

    .. math::

        2\langle \hat{f}'_{g,h}, \hat{\mu}'_{g,h;[p]}\rangle_{\mathcal{F}}-\langle \hat{f}'_{g,h}, \hat{T}_{C';[p,p]} \hat{f}'_{g,h}\rangle_{\mathcal{F}}  
         +\mu'\langle g,\hat{T}_A g\rangle_{\mathcal{G}} \\
        &\quad +
        2\langle \hat{f}_{g,h}, \hat{\mu}_{g,h;[q]}\rangle_{\mathcal{F}}-\langle \hat{f}_{g,h}, \hat{T}_{C;[q,q]} \hat{f}_{g,h}\rangle_{\mathcal{F}}  
        +\mu\langle h,\hat{T}_B h\rangle_{\mathcal{H}}.

    where 
    $\hat{\mu}'_{g,h;[p]}=\frac{1}{p}\Phi_{C';[p]}^* \vec{V}'_{g,h;[p]}$, 
    $\hat{\mu}_{g,h;[q]}=\frac{1}{q}\Phi_C^* \vec{V}_{g,h;[q]}$, and the covariance operators are defined analogously to :ref:`remark:max`. Hence by :ref:`remark:max` and the same argument as in :ref:`lemma:min`,

    .. math::

        \mathcal{E}(g,h)
        &=\frac{1}{p}(\vec{V}'_{g,h;[p]})^{\top} P_{C';[p,p]}\vec{V}'_{g,h;[p]}
         +\mu'\langle g,\hat{T}_A g\rangle_{\mathcal{G}}  +
        \frac{1}{q}\vec{V}_{g,h;[q]}^{\top}P_{C;[q,q]}\vec{V}_{g,h;[q]}
        +\mu\langle h,\hat{T}_B h\rangle_{\mathcal{H}}.

    2. Let $Y,G,H\in\mathbb{R}^n$ be defined with $G_i=g(A_i)$ and $H_i=h(B_i)$ as before. Now, let $\tilde{P}_{C'}=\frac{n}{p}I_{[p]}^{\top}P_{C';[p,p]}I_{[p]} \in \mathbb{R}^{n\times n}$ and
    $\tilde{P}_C=\frac{n}{q}I_{[q]}^{\top}P_{C';[q,q]}I_{[q]} \in \mathbb{R}^{n\times n}$. Then

    .. math::

        \frac{1}{p}(\vec{V}'_{g,h;[p]})^{\top} P_{C';[p,p]}\vec{V}'_{g,h;[p]} 
        &=\frac{1}{n}(Y^{\top}\tilde{P}_{C'} Y-2G^{\top}\tilde{P}_{C'}Y+G^{\top}\tilde{P}_{C'}G)\\
        \mu'\langle g,\hat{T}_A g\rangle_{\mathcal{G}} 
        &= \frac{\mu'}{n} G^{\top}G \\
        \frac{1}{q}\vec{V}_{g,h;[q]}^{\top}P_{C;[q,q]}\vec{V}_{g,h;[q]}
        &=\frac{1}{n}(H^{\top}\tilde{P}_CH-2G^{\top}\tilde{P}_CH+G^{\top}\tilde{P}_CG)\\ 
        \mu\langle h,\hat{T}_B h\rangle_{\mathcal{H}} 
        &=\frac{\mu}{n} H^{\top}H.

    Hereafter we use the same argument as in :ref:`lemma:min`.

Closed form - Estimator 3 (RKHS norm)
-------------------------------------

We study the RKHS-norm regularized *joint* estimator:

.. math::

    (\hat{g},\hat{h}) &= \arg \min _{g\in\mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_n\left[2\left\{g(A)-Y\right\} f'(C')-f'(C')^2\right]-\lambda'\|f'\|_\mathcal{F'}^2
     +\mu'\|g\|_\mathcal{G}^2 \\
    &\quad +
    \max_{f \in \mathcal{F}} \mathbb{E}_n\left[2\left\{h(B)-g(A)\right\} f(C)-f(C)^2\right] -\lambda\|f\|_\mathcal{F}^2  
    +\mu\|h\|_\mathcal{H}^2

.. lemma:: Formula of minimizers
   :name: lemma:min_4

    The minimizer takes the form $\hat{g} = \Phi_A^*\hat\alpha$, $\hat{h} = \Phi_B^*\hat\beta$ where,

    .. math::

        \hat{\beta} &= \left[ K_A \left\{ P_C + \left(P_{C'} K_A + P_C K_A + \mu'\right) \left( K_B P_C K_A \right)^{\dagger} \left( K_B P_C + \mu  \right)\right\} K_B \right]^{\dagger} K_A P_{C'} Y \\
        \hat{\alpha} &= \left( K_B P_C K_A \right)^{\dagger} \left( K_B P_C + \mu \right) K_B \hat{\beta}

    and

    .. math::

        P_C &= \left(K_C+\lambda\right)^{\dagger}K_C\\
        P_{C'} &= \left(K_{C'}+\lambda'\right)^{\dagger}K_{C'}

.. remark:: Subsetted estimator

    The subsetted estimator satisfies:

    .. math::

        \hat{\beta} &= \left[ K_A \left\{ \tilde{P}_C + \left(\tilde{P}_{C'} K_A + \tilde{P}_C K_A + \mu'\right) \left( K_B \tilde{P}_C K_A \right)^{\dagger} \left( K_B \tilde{P}_C + \mu  \right)\right\} K_B \right]^{\dagger} K_A \tilde{P}_{C'} Y \\
        \hat{\alpha} &= \left( K_B \tilde{P}_C K_A \right)^{\dagger} \left( K_B \tilde{P}_C + \mu \right) K_B \hat{\beta}

    with $\tilde{P}_{C'}=\frac{n}{p}I_{[p]}^{\top}P_{C';[p,p]}I_{[p]}$ and $\tilde{P}_{C}=\frac{n}{q}I_{[q]}^{\top}P_{C;[q,q]}I_{[q]}$. And

    .. math::

        P_{C';[p,p]} &= (K_{C';[p,p]}+\lambda I_{[p]}I_{[p]}^\top)^{-1}K_{C';[p,p]}, \quad K_{C';[p,p]}=I_{[p]}K_{C'}I_{[p]}^{\top} \\
        P_{C;[q,q]} &= (K_{C;[q,q]}+\lambda I_{[q]}I_{[q]}^\top)^{-1}K_{C;[q,q]}, \quad K_{C;[q,q]}=I_{[q]}K_{C}I_{[q]}^{\top}
