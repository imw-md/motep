# `motep apply`

This command calculates energies, forces, and stresses for the configurations written in
`data_in` using `potential_final` and write them in `data_out`.

## Usage

```
motep apply motep.apply.toml
```
where `motep.apply.toml` is
```toml
data_in = 'in.cfg'  # {'.cfg', '.xyz'}
data_out = 'out.cfg'  # {'.cfg', '.xyz'}
potential_final = 'final.mtp'

engine = 'numba'
```
