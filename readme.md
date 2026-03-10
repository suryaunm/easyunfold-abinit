The original easyunfold (https://github.com/SMTG-Bham/easyunfold) supports VASP and CASTEP only. This Git repo is extension of easy unfold to ABINIT.
Changes include: 
1. Addition of new file "abinit_wfk.py" -- to read the abinit generated wavefunction files in NETcdf format.
2. Modified "wavefun.py" — added AbinitWaveFunction wrapper class
3. Modified "unfold.py" - added 'abinit' as valid dft_code; fixed get_vbm_cbm for non-SCF occupancies
4. Modified "utils.py" - added 'abinit' to write_kpoints and read_kpoints dispatch.

Further details of the modifications are given below.
1. Changes in wavefun.py
1.1  New import
Added at the top of the file alongside existing imports:
from .abinit_wfk import AbinitWavefunction as WFKReader
1.2  New class: AbinitWaveFunction
A complete new class was added after CastepWaveFunction. It wraps AbinitWavefunction (the NetCDF reader) and exposes the same interface as VaspWaveFunction and CastepWaveFunction so that all three codes are interchangeable inside unfold.py.
Every property and method delegates to the underlying reader:
Property / Method	Implementation	Notes
kpoints	self.wfc._kvecs	shape (nkpt, 3) fractional
nspins	self.wfc._nspin	int
mesh_size	self.wfc._ngrid	None if not in file
bands	self.wfc._bands	shape (nspin, nkpt, nbands) eV
occupancies	self.wfc._occs	shape (nspin, nkpt, nbands)
fermie	self.wfc._fermie	float eV — NEW property not in VASP/CASTEP
get_gvectors(ik)	self.wfc.get_gvectors(ik)	shape (nplw, 3) int, 1-based ik
get_band_coeffs	self.wfc.read_band_coeffs	shape (nplw,) complex, 1-based indices
The class also provides a from_file classmethod:
  @classmethod
  def from_file(cls, fname: str, lsorbit: bool = False):
      return cls(WFKReader(fnm=fname, lsorbit=lsorbit))

4. Changes in unfold.py
4.1  Unfold.__init__ — added abinit branch
Original code (lines ~805–810):
  if dft_code == 'vasp':
      self.wfc = VaspWaveFunction(Wavecar(...))
  elif dft_code == 'castep':
      self.wfc = CastepWaveFunction.from_file(fname)
  else:
      raise NotImplementedError(f'Code {dft_code} has not being implemented!')

Updated code — added abinit branch before the else clause:
  if dft_code == 'vasp':
      self.wfc = VaspWaveFunction(Wavecar(...))
  elif dft_code == 'castep':
      self.wfc = CastepWaveFunction.from_file(fname)
  elif dft_code == 'abinit':                            # ← NEW
      self.wfc = AbinitWaveFunction.from_file(fname, lsorbit=self._lsoc)
  else:
      raise NotImplementedError(f'Code {dft_code} has not being implemented!')
4.2  get_vbm_cbm — fixed for non-SCF Abinit runs
In a non-SCF band structure run (iscf = -2), Abinit writes occupancies of 2.0 for all bands at all k-points because it reads them from the density, not from self-consistent iteration. This means the original VBM/CBM detection (which looks for bands where occ drops from non-zero to zero) fails — either VBM or CBM is nonsensical.
Original code:
  def get_vbm_cbm(self, thresh=1e-8):
      occ = self.wfc.occupancies
      occupied = np.abs(occ) > thresh
      vbm = float(self.bands[occupied].max())
      cbm = float(self.bands[~occupied].min())
      return vbm, cbm

Updated code — falls back to the Fermi energy from the WFK file when all bands are occupied:
  def get_vbm_cbm(self, thresh=1e-8):
      occ = self.wfc.occupancies
      occupied = np.abs(occ) > thresh
      if not occupied.any() and hasattr(self.wfc, 'fermie'):
          fermie = self.wfc.fermie
          if fermie is not None:
              return fermie, fermie   # use Fermi energy as reference
      if occupied.any() and (~occupied).any():
          return float(self.bands[occupied].max()),
                 float(self.bands[~occupied].min())
      nb = self.bands.shape[-1]
      return float(self.bands[:,:,:nb//2].max()),
             float(self.bands[:,:,nb//2:].min())
 
5. Changes in utils.py
5.1  write_kpoints — added abinit dispatch
Original: raises NotImplementedError for any code that is not 'vasp' or 'castep'.
Updated: added abinit case that writes kpoints in VASP KPOINTS format (which the user then converts for the Abinit input file):
  if code == 'abinit':
      return write_kpoints_vasp(kpoints, outpath, *args, **kwargs)
5.2  read_kpoints — added abinit dispatch
Same pattern: added abinit case that reads VASP-format KPOINTS files:
  if code == 'abinit':
      return read_kpoints_vasp(path, **kwargs)

6. New File: abinit_wfk.py
This is a completely new file. It provides the AbinitWavefunction class that reads Abinit NetCDF WFK files. The public interface deliberately mirrors the Wavecar class.
6.1  Key design decisions
Design choice	Reason
Reads everything on __init__, closes file immediately	Matches Wavecar behaviour; object is self-contained after construction
_read_gvectors: tries file first, reconstructs if sentinels found	Abinit often writes INT32_MIN (-2147483647) as placeholder. Reconstruction uses the same KE filter as VASP's get_gvectors but in Hartree/Bohr atomic units
_read_coefficients: uses np.ma.filled instead of np.array	netCDF4-python returns MaskedArray. np.array() silently fills masks with 9.97e36. np.ma.filled(..., 0.0) correctly replaces padding with zero while preserving real data. Raises ValueError with clear message if ALL values are masked
fermie property	Non-SCF runs have all occupancies = 2.0 so VBM/CBM detection fails. Reading fermi_energy from the WFK directly (in Hartree, converted to eV) provides a reliable energy reference
G-vector reconstruction algorithm	Identical to VASP wavecar.py get_gvectors but uses: ecut in Hartree, KE = 0.5*|2pi*(G+k)*inv(rprim).T|^2, sorts by ascending KE for deterministic ordering, verifies count matches number_of_coefficients from file
6.2  Important constants
  HARTREE_TO_EV = 27.211386245988
  BOHR_TO_ANG   = 0.529177210903
  TPI           = 2.0 * np.pi
6.3  Private methods
Method	Description
_read_header()	Reads primitive_vectors, k-points, nspin, nkpts, nbands, nspinor, ecut, ngrid
_read_bands()	Reads eigenvalues (Ha→eV), occupations, nplws, kpath, fermie
_read_gvectors()	Reads or reconstructs G-vectors for all kpoints
_read_coefficients()	Reads complex coefficients via np.ma.filled; shape (nspin,nkpt,nbands,nspinor,nplw)
6.4  Public methods (mirror Wavecar interface)
Method	Description
get_gvectors(ikpt)	Returns (nplw, 3) int array. ikpt is 1-based
read_band_coeffs(ispin, ikpt, iband, norm)	Returns (nplw,) complex array. All indices 1-based. SOC: returns (2*nplw,)
is_soc()	True if nspinor==2
is_gamma()	Always False (Abinit has no gamma-only mode)

8. Complete Step-by-Step Workflow
Step 1 — Install the modified easyunfold files
Copy all modified files to the easyunfold installation directory:
  SITE=$(python3 -c 'import easyunfold; import os; print(os.path.dirname(easyunfold.__file__))')
  cp abinit_wfk.py wavefun.py unfold.py utils.py $SITE/
  find $SITE -name '*.pyc' -delete

Step 2 — Initialise easyunfold
Create a POSCAR files for primitive cell and supercell cell, high-symmetry kpath in primitive cell. 
CLI commad: easyunfold generate prim.POSCAR sc.POSCAR kpath.txt --code abinit
Important: Donot forget to add --code abinit as this is the only way the easyunfold knows the code we are using and the kpath is saved in the "easyunfold.json" file. This "easyunfold.json" file will have all the information and is needed to be present in the same folder where you perform the unfolding. This creates easyunfold.json (stores M, k-path, metadata) and KPOINTS (the supercell kpoints to compute in Abinit).

Step 3 — Convert KPOINTS to Abinit format -- optinal or you can copy paste the kpoints into input file
The KPOINTS file is in VASP format. Extract the kpoints for the Abinit input:
  python3 - << 'EOF'
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
  EOF
Copy the output into trf2_2.abi.

Step 4 — Run Abinit SCF (Run 1)
generates trf2_1o_DEN
Step 5 — Run Abinit non-SCF band structure (Run 2)
  For calculation of eigen energies, 
  Specify the kpts as follow:
  "
  iscf       -2     #a non-self-consistent calculation for eigen value computation
  getden     -1     #get change density from precious run, the densityf file name is inputname_1i_DEN
  kptopt      0     #read directly nkpt, kpt
  nkpt        x     #number of kpts
  kpt    list[x]    #list kpts
  tolwfr   1.0d-12
  nstep      50
  prtden      1
  istwfk     *1     #do NOT take advantage of the time-reversal symmetry
  iomode      3     #Use NetCDF library to produce files according to the ETSF specification
  prteig      1
  prtwf       1
  "  
  
This generates trf2_2o_WFK.nc. Verify coefficients were written:
  python3 -c "
  import netCDF4 as nc, numpy as np
  ds = nc.Dataset('trf2_2o_WFK.nc','r')
  raw = ds.variables['coefficients_of_wavefunctions']
  filled = np.ma.filled(raw[0,5,4,0,:3,0], 0.0)
  masked = np.ma.is_masked(raw[0,5,4,0,:3,0])
  print('masked:', masked, '  values:', filled)
  ds.close()"
Expected output: masked: False with non-zero, non-uniform values.

Step 7 — Calculate spectral weights
  rm -f easyunfold.json.bak  # optional backup
  easyunfold unfold calculate trf2_2o_WFK.nc
This reads the WFK, computes the spectral weight P(K,E) for each supercell kpoint by projecting wavefunction coefficients onto the primitive cell G-sphere, and saves results to easyunfold.json.

Step 8 — Plot the unfolded band structure
  easyunfold unfold plot \
      --eref <fermi_energy_in_eV> \
      --emin -5 --emax 9 \
      --intensity 10 \
      -o unfolded_bands.png
The --eref value should be the Fermi energy from your SCF run (printed in trf2_1.abo or readable from the DEN file). For Mo, this is approximately 17.0 eV from our runs.
To find the Fermi energy automatically from the WFK:
  python3 -c "
  import netCDF4 as nc, numpy as np
  ds = nc.Dataset('trf2_2o_WFK.nc','r')
  ef = float(np.asarray(ds.variables['fermi_energy']).flat[0]) * 27.211386
  print(f'Fermi energy: {ef:.6f} eV')
  ds.close()"

10. . Troubleshooting
Symptom	Cause	Fix
coefficients_of_wavefunctions all masked	Abinit compiled with MPI-IO, or prtwf not set	Recompile Abinit without MPI-IO. Add prtwf 1 and iomode 3 to Run 2 input
G-vector count mismatch error	istwfk not set to 1 for all kpoints	Add istwfk *1 to the input. This forces full G-sphere (no time-reversal compression)
All spectral weights uniform (~1/nplw)	G-vector ordering mismatch between reconstruction and Abinit internal order	The KE-sorted reconstruction should match. Verify ecut matches between runs
VBM/CBM nonsensical (e.g. CBM < VBM)	All occupancies = 2.0 in non-SCF run	Fixed by get_vbm_cbm fallback to fermie. Use --eref explicitly in plot command
Plot shows all bands equally bright	Old easyunfold.json cached from broken run	Delete easyunfold.json and rerun easyunfold unfold calculate
kpoints mismatch error from easyunfold	kptopt not set to 0, or kpt list has wrong kpoints	Use kptopt 0 with the exact kpoints from the KPOINTS file generated by easyunfold
