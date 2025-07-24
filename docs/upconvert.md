# `motep upconvert`

This command up-converts an MTP potential with a higher level,
a larger radial basis size, and/or more species.
This enables us to retrain the potential with higher flexibility.

## Usage

```bash
motep upconvert  # The default setting below is applied.
```
or
```bash
motep upconvert upconvert.toml
```
where `upconvert.toml` is
```toml
[potentials]
base = 'base.mtp'  # trained potential with, e.g., a lower level
initial = 'initial.mtp'  # untrained potential with, e.g., a higher level
final = 'final.mtp'  # upconverted potential
```
