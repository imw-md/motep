Loss function
=============

The following shows the loss function to be optimized during the training.

Definition
----------

.. math::

    L \equiv
    w_\mathrm{e} L_\mathrm{e} +
    w_\mathrm{f} L_\mathrm{f} +
    w_\mathrm{s} L_\mathrm{s}

- :math:`L_\mathrm{e}`: contribution from energy

  - ``energy_per_atom = true``:

    .. math::
        L_\mathrm{e} \equiv \sum_{k=1}^{N_\mathrm{conf}}
        (\hat{E}_k - \hat{E}_k^\mathrm{ref})^2

    where

    .. math::
        \hat{E}_k \equiv \frac{E_k}{N_{\mathrm{atom},k}}

- :math:`L_\mathrm{f}`: Contribution from forces

  - ``forces_per_atom = true``:

    .. math::
        L_\mathrm{f} \equiv \sum_{k=1}^{N_\mathrm{conf}}
        \frac{1}{N_{\mathrm{atom},k}} \sum_{i=1}^{N_{\mathrm{atom},k}} \sum_{\alpha=1}^{3}
        (F_{k,i\alpha} - F_{k,i\alpha}^\mathrm{ref})^2

- :math:`L_\mathrm{s}`: Contribution from stress

  - ``stress_times_volume = true``:

    .. math::
        L_\mathrm{s} \equiv \sum_{k=1}^{N_\mathrm{conf}} V_k^2
        \sum_{\alpha=1}^{3} \sum_{\beta=1}^{3}
        (\sigma_{k,\alpha\beta} - \sigma_{k,\alpha\beta}^\mathrm{ref})^2

    ``energy_per_atom`` is also respected.

Default
-------

``motep.toml``

.. code-block:: toml

    [loss]
    energy_weight = 1.0
    forces_weight = 0.01
    stress_weight = 0.001
    energy_per_atom = true
    forces_per_atom = true
    stress_times_volume = true
    energy_per_conf = true
    forces_per_conf = true
    stress_per_conf = true
