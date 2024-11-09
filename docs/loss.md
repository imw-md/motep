# Loss function

## Definition

$$
L \equiv
w_\mathrm{e} L_\mathrm{e} + w_\mathrm{f} L_\mathrm{f} + w_\mathrm{s} L_\mathrm{s}
$$

- $L_\mathrm{e}$: contribution from energy
  - `energy_per_atom = true`
    $$
    L_\mathrm{e} \equiv \sum_{k=1}^{N_\mathrm{conf}}
    (\hat{E}_k - \hat{E}_k^\mathrm{ref})^2
    $$
    where
    $$
    \hat{E}_k \equiv \frac{E_k}{N_{\mathrm{atom},k}}
    $$

- $L_\mathrm{f}$: contribution from forces
  - `forces_per_atom = true`
    $$
    L_\mathrm{f} \equiv \sum_{k=1}^{N_\mathrm{conf}}
    \frac{1}{N_{\mathrm{atom},k}} \sum_{i=1}^{N_{\mathrm{atom},k}} \sum_{\alpha=1}^{3}
    (F_{k,i\alpha} - F_{k,i\alpha}^\mathrm{ref})^2
    $$

- $L_\mathrm{s}$: contribution from stress
  - `stress_times_volume = false`
    $$
    L_\mathrm{s} \equiv \sum_{k=1}^{N_\mathrm{conf}}
    \sum_{\alpha=1}^{3} \sum_{\beta=1}^{3}
    (\sigma_{k,\alpha\beta} - \sigma_{k,\alpha\beta}^\mathrm{ref})^2
    $$
  - `stress_times_volume = true`:
    The stress is scaled by the cell volume, making it an extensive property.

## Recipes

### Default

```toml
[loss]
energy_weight = 1.0
forces_weight = 0.01
stress_weight = 0.001
energy_per_atom = true
forces_per_atom = true
stress_times_volume = false
```
