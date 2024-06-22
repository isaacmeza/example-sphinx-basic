Semiparametric Estimation
=========================

.. toctree::
   :maxdepth: 2

The goal is general purpose learning and inference for the longitudinal causal parameter :math:`\theta_0 \in \mathbb{R}`. 
Section 3 demonstrates that many such parameters :math:`\theta_0` have multiply robust moment functions with nuisance parameters 
:math:`(\nu_0, \delta_0, \alpha_0, \eta_0)`. In this section, I describe a meta algorithm to turn estimators 
:math:`(\hat{\nu}, \hat{\delta}, \hat{\alpha}, \hat{\eta})` into an estimator :math:`\hat{\theta}` such that 
:math:`\hat{\theta}` has a valid and practical confidence interval. This meta algorithm can be seen as an extension of classic 
one step corrections [66] amenable to the use of modern machine learning, and it has been termed debiased machine learning [23]. 
Unlike targeted minimum loss inference with a finite sample, it does not involve substitution, iteration, or bootstrapping [91][18].

The target estimator :math:`\hat{\theta}` as well as its confidence interval will depend on nuisance estimators 
:math:`(\hat{\nu}, \hat{\delta}, \hat{\alpha}, \hat{\eta})`. I refrain from instantiating these estimators until Section 7. 
As we will see in subsequent analysis, the general theory only requires that each nuisance estimator converges to the corresponding 
nuisance parameter in mean square error. The general meta algorithm is as follows.

**Algorithm 5.1 (Debiased machine learning).** Given a sample :math:`(Y_i, W_i)` 
(:math:`i = 1, \ldots, n`), partition the sample into folds (:math:`I_\ell`) (:math`\ell = 1, \ldots, L`). 
Denote by :math:`I^c_\ell` the complement of :math:`I_\ell`.

1. For each fold :math:`\ell`, estimate :math:`(\hat{\nu}_\ell, \hat{\delta}_\ell, \hat{\alpha}_\ell, \hat{\eta}_\ell)` 
   from observations in :math:`I^c_\ell`.

2. Estimate :math:`\theta_0` as

   .. math::

      \hat{\theta} = \frac{1}{n} \sum_{\ell=1}^L \sum_{i \in I_\ell} 
      \left[ \hat{\nu}_\ell(W_i) + \hat{\alpha}_\ell(W_i)\{Y_i - \hat{\delta}_\ell(W_i)\} + \hat{\eta}_\ell(W_i)\{\hat{\delta}_\ell(W_i) - \hat{\nu}_\ell(W_i)\} \right].

3. Estimate its :math:`(1 - \alpha)100%` confidence interval as :math:`\hat{\theta} \pm c_\alpha \hat{\sigma} n^{-1/2}`, 
   where :math:`c_\alpha` is the :math:`1 - \alpha/2` quantile of the standard Gaussian and

   .. math::

      \hat{\sigma}^2 = \frac{1}{n} \sum_{\ell=1}^L \sum_{i \in I_\ell} 
      \left[ \hat{\nu}_\ell(W_i) + \hat{\alpha}_\ell(W_i)\{Y_i - \hat{\delta}_\ell(W_i)\} + \hat{\eta}_\ell(W_i)\{\hat{\delta}_\ell(W_i) - \hat{\nu}_\ell(W_i)\} - \hat{\theta} \right]^2.


   semiparametric/Mediation Analysis
   semiparametric/Long Term Effect





