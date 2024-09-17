import pathlib

import ase.io

from motep.io.mlip.cfg import write_cfg

root = pathlib.Path("/Users/ikeda/Documents/projects/2022/CCSD_MLIP_AF/data")
molecules = [
    762,  # H2
    291,  # CH4  # due to some reasons level 4 cannot be fitted
    # 236,  # C6H6
    14214,  # HF
    23208,  # O3
]
nsteps = {762: 50, 291: 50, 236: 50, 14214: 50, 23208: 50}
for molecule in molecules:
    print(molecule)
    fn = root / f"tblite-ase/md/ChemSpider_3D/{molecule}/ase.xyz"
    step = nsteps[molecule]
    images = ase.io.read(fn, index=':')[::step]
    dn = pathlib.Path(str(molecule))
    dn.mkdir(parents=True, exist_ok=True)
    write_cfg(dn / "training.cfg", images, key_energy="energy")
