# Loss function

## Definition

$$
L = w_\mathrm{e} L_\mathrm{e} + w_\mathrm{f} L_\mathrm{f} + w_\mathrm{s} L_\mathrm{s}
$$

## Recipes

### Default

```toml
[loss]
energy_weight = 1.0
forces_weight = 0.01
stress_weight = 0.001
stress_times-volume = false
```

- `stress_times_volume`: If `true`, the stress is scaled by the cell volume,
making it an extensive property.
