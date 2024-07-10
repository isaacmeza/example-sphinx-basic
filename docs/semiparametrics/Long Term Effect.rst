Long-term Effect Analysis
==========================

Let $X \in \mathbb{R}^{p}$ be baseline covariates. Let $D \in \{0, 1\}$ indicate treatment assignment. Let $M \in \mathbb{R}$ be an intermediate/short-term outcome and $Y \in \mathbb{R}$ be a long-term outcome. An analyst may wish to measure the effect of $D$ on $Y$ (a long-term outcome), yet the experimental sample only includes $M$ (a short-term outcome).

If the analyst has access to an additional observational sample that includes the long-term outcome, then long-term causal inference is still possible. Specifically, assume the analyst has access to (i) an experimental sample, indicated by $G=0$, where $(D, M, X)$ are observed; and (ii) an observational sample, indicated by $G=1$, where $(M, X, Y)$ are observed, and $D$ is either observed or not. Depending on whether $D$ is also revealed in the observational sample will give rise to different assumptions that identify the long-term treatment effect. Specifically, the key identifying assumption when we do not observe $D$ is that the short-term outcome is a statistical surrogate for the long-term outcome, while the identifying assumption for the case when we observe $D$ is that unobserved confounding is mediated through the short-term outcome in the observational sample. Following [Athey et al., 2020b](https://arxiv.org/abs/1603.09326), we refer to these models as \emph{Surrogacy model} or \emph{Latent unconfounded model}, respectively ([Athey et al., 2020a](https://arxiv.org/abs/2006.09676)).

.. admonition:: Long-term effect

   Formally, define the long-term counterfactual $\mathbb{E}\left[Y^{(d)}\right]$ as the counterfactual mean outcome for the full population in the thought experiment in which everyone is assigned treatment value $D=d$.

The long-term effect defined for the experimental or observational subpopulation is similar, introducing the fixed local weighting $\ell(G)=\mathds{1}_{G=0} / \mathbb{P}(G=0)$ or $\ell(G)=\mathds{1}_{G=1} / \mathbb{P}(G=1)$, respectively.

Surrogacy Model
----------------

**Surrogacy model** Define the regression and the conditional distribution

.. math::
   \begin{aligned}
   \gamma_{0}(m, x, g) & = \mathbb{E}[Y \mid M=m, X=x, G=g] \\
   \mathbb{P}(m \mid d, x, g) & = \mathbb{P}(M=m \mid D=d, X=x, G=g)
   \end{aligned}

Then the long-term counterfactual is

.. math::
   \operatorname{LONG}(d) = \mathbb{E}\left\{\int \gamma_{0}(m, X, 1) \mathrm{d} \mathbb{P}(m \mid d, X, 0)\right\}

In summary, $W=(D, M, X, G)$, $W_{1}=(D, X, G)$, and $W_{2}=M$. Moreover,

.. math::
   m\left(W_{1}, w_{2}, \gamma_{0}\right) = \gamma_{0}(m, X, 1), \quad \mathbb{Q}\left(w_{2} \mid W_{1}\right) = \mathbb{P}(m \mid d, X, 0)


.. math::
   \begin{aligned}
   \gamma_{0}(m, x, g) & = \mathbb{E}[Y \mid M=m, X=x, G=g] \\
   \mathbb{P}(m \mid d, x, g) & = \mathbb{P}(M=m \mid D=d, X=x, G=g)
   \end{aligned}

and

.. math::
   m\left(W_{1}, w_{2}, \gamma_{0}\right) = \gamma_{0}(m, X, 1), \quad \mathbb{Q}\left(w_{2} \mid W_{1}\right) = \mathbb{P}(m \mid d, X, 0)

Take

.. math::
   \begin{aligned}
   \nu_{0}(W) & = \int \gamma_{0}(m, X, 1) \mathrm{d} \mathbb{P}(m \mid d, X, 0) \\
   \delta_{0}(W) & = \gamma_{0}(M, X, 1) \\
   \alpha_{0}(W) & = \frac{\mathds{1}_{G=1}}{\mathbb{P}(G=1 \mid M, X)} \frac{\mathbb{P}(d \mid M, X, G=0) \mathbb{P}(G=0 \mid M, X)}{\mathbb{P}(d \mid X, G=0) \mathbb{P}(G=0 \mid X)} \\
   \eta_{0}(W) & = \frac{\mathds{1}_{G=0} \mathds{1}_{D=d}}{\mathbb{P}(d \mid X, G=0) \mathbb{P}(G=0 \mid X)}
   \end{aligned}

Denote the treatment propensity scores

.. math::
   \pi_{0}(d \mid X, G=0) = \mathbb{P}(D=d \mid X, G=0), \quad \rho_{0}(d \mid M, X, G=0) = \mathbb{P}(D=d \mid M, X, G=0)

and the selection propensity scores

.. math::
   \pi_{0}^{\prime}(g \mid X) = \mathbb{P}(G=g \mid X), \quad \rho_{0}^{\prime}(g \mid M, X) = \mathbb{P}(G=g \mid M, X)

Latent Unconfounded Model
-------------------------

**Latent unconfounded model** When we observe $D$ in the observational sample, the regression becomes

.. math::
   \gamma_{0}(m, x, g, d)  = \mathbb{E}[Y \mid M=m, X=x, G=g, D=d]

and the long-term counterfactual becomes

.. math::
   \operatorname{LONG}(d) = \mathbb{E}\left\{\int \gamma_{0}(m, X, 1, d) \mathrm{d} \mathbb{P}(m \mid d, X, 0)\right\}




Under this model we have that

.. math::
   \begin{aligned}
   \gamma_{0}(m, x, g, d) & = \mathbb{E}[Y \mid M=m, X=x, G=g, D=d] \\
   \mathbb{P}(m \mid d, x, g) & = \mathbb{P}(M=m \mid D=d, X=x, G=g)
   \end{aligned}

and

.. math::
   m\left(W_{1}, w_{2}, \gamma_{0}\right) = \gamma_{0}(m, X, 1, d), \quad \mathbb{Q}\left(w_{2} \mid W_{1}\right) = \mathbb{P}(m \mid d, X, 0)

Take

.. math::
   \begin{aligned}
   \nu_{0}(W) & = \int \gamma_{0}(m, X, 1, d) \mathrm{d} \mathbb{P}(m \mid d, X, 0) \\
   \delta_{0}(W) & = \gamma_{0}(M, X, 1, d) \\
   \alpha_{0}(W) & = \frac{\mathds{1}_{G=1}\mathds{1}_{D=d}}{\mathbb{P}(G=1 \mid M, X, D=d)} \frac{\mathbb{P}(G=0 \mid M, X, D=d)}{\mathbb{P}(D=d \mid X, G=0) \mathbb{P}(G=0 \mid X)} \\
   \eta_{0}(W) & = \frac{\mathds{1}_{G=0} \mathds{1}_{D=d}}{\mathbb{P}(D=d \mid X, G=0) \mathbb{P}(G=0 \mid X)}
   \end{aligned}

Denote the treatment propensity score

.. math::
   \pi_{0}(d \mid X, G=g) = \mathbb{P}(D=d \mid X, G=g)

and the selection propensity scores

.. math::
   \pi_{0}^{\prime}(g \mid X) = \mathbb{P}(G=g \mid X), \quad \rho^{\prime}_{0}(g \mid M, X, D=d) = \mathbb{P}(G=g \mid M, X, D=d)

