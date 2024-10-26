# Loss function

## Definition

$$
L = w_\mathrm{e} L_\mathrm{e} + w_\mathrm{f} L_\mathrm{f} + w_\mathrm{s} L_\mathrm{s}
$$

## Recipes

### Default

```toml
[loss]
energy-weight = 1.0
force-weight = 0.01
stress-weight = 0.001
```
