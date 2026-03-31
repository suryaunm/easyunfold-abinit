# easyunfold — ABINIT Extension

> The original [easyunfold](https://github.com/SMTG-Bham/easyunfold) supports VASP and CASTEP.  
> This repository extends its capabilities to fully support **ABINIT** calculations.

---

## 🚀 Quick Start & Workflow

### 1. Install the Modified Files

Copy the modified files into your local easyunfold installation directory:

```bash
SITE=$(python3 -c 'import easyunfold; import os; print(os.path.dirname(easyunfold.__file__))')
cp abinit_wfk.py wavefun.py unfold.py utils.py $SITE/
find $SITE -name '*.pyc' -delete
```

---

### 2. Initialise easyunfold

### Load python --- easley
```bash
module purge
module load python/3.13.5-ajzf

source ~/scratch-easley/codes/python/suryab/bin/activate

```

### Load python --- hopper
```bash
module purge
module load gcc/12.1.0-crtl
module load python/3.13.2-tvzw

source ~/codes/suryab/bin/activate

```


Generate the `easyunfold.json` configuration and the supercell k-points file (`KPOINTS`).  
You will need POSCAR files for both your primitive cell and supercell, along with a high-symmetry k-path text file.

```bash
easyunfold generate prim.POSCAR sc.POSCAR kpath.txt --code abinit
```

> ⚠️ **Important:** You must include the `--code abinit` flag. This tells easyunfold which DFT code to configure and saves this preference in `easyunfold.json`, which must remain in your working directory.

---

### 3. Convert KPOINTS to ABINIT Format

The generated `KPOINTS` file is in VASP format. Use this Python script to extract and format the k-points for your ABINIT input:

```python
kpts = []
with open('KPOINTS') as f:
    lines = f.readlines()
    nk = int(lines[1])
    for line in lines[3:3+nk]:
        k = line.split()[:3]
        kpts.append(f'  {k[0]}  {k[1]}  {k[2]}')

print(f'nkpt  {len(kpts)}')
print('kpt')
print('\n'.join(kpts))
print(f'wtk  {len(kpts)}*1.0')
```

Copy the output into your non-SCF ABINIT input file (e.g., `trf2_2.abi`).

---

### 4. Run ABINIT Calculations

#### Run 1 — SCF Calculation

Run a standard self-consistent field (SCF) calculation to generate the density file (e.g., `trf2_1o_DEN`).

#### Run 2 — Non-SCF Band Structure

Run a non-self-consistent calculation to compute eigenvalues at the easyunfold k-points.  
Ensure your input file contains the following parameters:

```
iscf       -2        # Non-self-consistent calculation for eigenvalue computation
getden     -1        # Get charge density from the previous run
kptopt      0        # Read directly nkpt and kpt lists
nkpt        x        # Number of k-points from Step 3
kpt   list[x]        # List of k-points from Step 3
tolwfr      1.0d-12
nstep       50
prtden      1
istwfk     *1        # REQUIRED: Do NOT use time-reversal symmetry
iomode      3        # REQUIRED: NetCDF output (ETSF specification)
prteig      1
prtwf       1        # REQUIRED: Write wavefunction coefficients
```

This generates the wavefunction file: `trf2_2o_WFK.nc`.

<details>
<summary>✅ Optional: Verify coefficients were written correctly</summary>

```python
import netCDF4 as nc, numpy as np

ds = nc.Dataset('trf2_2o_WFK.nc', 'r')
raw = ds.variables['coefficients_of_wavefunctions']
filled = np.ma.filled(raw[0, 5, 4, 0, :3, 0], 0.0)
masked = np.ma.is_masked(raw[0, 5, 4, 0, :3, 0])
print('masked:', masked, '  values:', filled)
ds.close()
```

**Expected output:** `masked: False` followed by non-zero, non-uniform values.

</details>

---

### 5. Calculate Spectral Weights

Read the WFK file, compute the spectral weight $P(K,E)$ for each supercell k-point by projecting wavefunction coefficients onto the primitive cell G-sphere, and save the results.

```bash
rm -f easyunfold.json.bak   # optional backup
easyunfold unfold calculate trf2_1o_WFK.nc
```

---

### 6. Plot the Unfolded Band Structure

```bash
easyunfold unfold plot --emin -5 --emax 9 --intensity 10 --eref -o .png
```

<details>
<summary>💡 Finding the Fermi Energy</summary>

The `--eref` value should be the Fermi energy from your SCF run (printed in `trf2_1.abo` or read from the DEN file). To extract it automatically from the WFK:

```python
import netCDF4 as nc, numpy as np

ds = nc.Dataset('trf2_2o_WFK.nc', 'r')
ef = float(np.asarray(ds.variables['fermi_energy']).flat[0]) * 27.211386
print(f'Fermi energy: {ef:.6f} eV')
ds.close()
```

</details>

---

## 🛠️ Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `coefficients_of_wavefunctions` all masked | ABINIT compiled with MPI-IO, or `prtwf` not set | Recompile ABINIT without MPI-IO. Add `prtwf 1` and `iomode 3` to the Run 2 input |
| G-vector count mismatch error | `istwfk` not set to `1` for all k-points | Add `istwfk *1` to the input to force the full G-sphere (no time-reversal compression) |
| All spectral weights uniform (~1/nplw) | G-vector ordering mismatch between reconstruction and internal order | Verify `ecut` matches between runs |
| VBM/CBM nonsensical (e.g. CBM < VBM) | All occupancies = 2.0 in non-SCF run | Handled by `get_vbm_cbm` fallback to `fermie`. Use `--eref` explicitly in the plot command |
| Plot shows all bands equally bright | Old `easyunfold.json` cached from a broken run | Delete `easyunfold.json` and rerun `easyunfold unfold calculate` |
| K-points mismatch error | `kptopt` not set to `0`, or wrong k-point list | Use `kptopt 0` with the exact k-points from the `KPOINTS` file generated by easyunfold |

---

## 🔍 Under the Hood: Code Modifications

### New File: `abinit_wfk.py`

Provides the `AbinitWavefunction` class to read ABINIT NetCDF WFK files. The public interface deliberately mirrors the `Wavecar` class.

**Key design decisions:**

- **Self-contained initialisation** — reads everything on `__init__` and closes the file immediately, matching `Wavecar` behaviour.
- **G-vector reconstruction** — tries to read G-vectors from the file first; falls back to reconstruction from `ecut` if ABINIT sentinel placeholders (`INT32_MIN`) are found. Uses the same KE filter as VASP's `get_gvectors` in Hartree/Bohr atomic units, sorted by ascending KE for deterministic ordering.
- **Safe coefficient reading** — uses `np.ma.filled(..., 0.0)` instead of `np.array` to correctly handle the `MaskedArray` returned by netCDF4-python. Raises a clear `ValueError` with remediation instructions if all coefficients are masked.
- **`fermie` property** — non-SCF runs force all occupancies to 2.0, making standard VBM/CBM detection unreliable. `fermi_energy` is read directly from the WFK and converted from Hartree to eV.

---

### `wavefun.py` Modifications

- Added import: `from .abinit_wfk import AbinitWavefunction as WFKReader`
- Added `AbinitWaveFunction` wrapper class that delegates every property (`kpoints`, `nspins`, `mesh_size`, `bands`, `occupancies`, `fermie`) and method (`get_gvectors`, `get_band_coeffs`) to the underlying reader, making VASP, CASTEP, and ABINIT fully interchangeable inside `unfold.py`.

---

### `unfold.py` Modifications

- **`__init__`** — added the `abinit` branch to route wavefunction loading through `AbinitWaveFunction.from_file(fname, lsorbit=self._lsoc)`.
- **`get_vbm_cbm`** — updated VBM/CBM detection to fall back to the WFK Fermi energy when all bands report as occupied (a quirk of ABINIT `iscf = -2` runs).

---

### `utils.py` Modifications

- **`write_kpoints`** — added `abinit` dispatch that writes k-points in VASP `KPOINTS` format for user adaptation.
- **`read_kpoints`** — added `abinit` dispatch to read VASP-format `KPOINTS` files.
