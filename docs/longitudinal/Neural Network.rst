.. _neural-networks:

Neural Networks
===============

We now consider the case where the function classes correspond to neural networks. In such case, the (joint) estimator takes the form:

.. math::

    (\hat{g},\hat{h}) = \arg \min _{\theta_1,\theta_2} 
    \max_{\omega_1, \omega_2} \left\{ \mathbb{E}_n\left[2\left\{g_{\theta_1}(A)-Y\right\} f_{\omega_1}'(C')-f_{\omega_1}'(C')^2\right]
     +\mu'\E_n\{g_{\theta_1}(A)^2\} \\
    \quad + \mathbb{E}_n\left[2\left\{h_{\theta_2}(B)-g_{\theta_1}(A)\right\} f_{\omega_2}(C)-f_{\omega_2}(C)^2\right]   
    +\mu\E_n\{h_{\theta_2}(B)^2\}\right\}

where :math:`\theta_1, \theta_2, \omega_1,\omega_2` are weights of the neural networks.

We use the Optimistic Adam algorithm of :cite:`Daskalakis` to solve the previous minimax problem as was also proposed in :cite:`dikkala2020minimax`.

.. remark:: Subsetted estimator

    Modify the computation of the loss for the adversary to be zero for the observations outside the restriction:

    .. code-block:: python

        test = self.adversary(zb)
        test[indices_] = 0 
        G_loss = - torch.mean((yb - pred) * test) + torch.mean(test**2)

.. bibliography::

    @article{Daskalakis,
      author       = {Constantinos Daskalakis and
                      Andrew Ilyas and
                      Vasilis Syrgkanis and
                      Haoyang Zeng},
      title        = {Training GANs with Optimism},
      journal      = {CoRR},
      volume       = {abs/1711.00141},
      year         = {2017},
      url          = {http://arxiv.org/abs/1711.00141},
      eprinttype   = {arXiv},
      eprint       = {1711.00141},
      timestamp    = {Mon, 13 Aug 2018 16:47:50 +0200},
      biburl       = {https://dblp.org/rec/journals/corr/abs-1711-00141.bib},
      bibsource    = {dblp computer science bibliography, https://dblp.org}
    }

    @article{dikkala2020minimax,
      title       = {Minimax estimation of conditional moment models},
      author      = {Dikkala, Nishanth and Lewis, Greg and Mackey, Lester and Syrgkanis, Vasilis},
      journal     = {Advances in Neural Information Processing Systems},
      volume      = {33},
      pages       = {12248--12262},
      year        = {2020}
    }
