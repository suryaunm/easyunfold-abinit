"""
Code for reading eigenvalues and plane wave coefficients from an Abinit
NetCDF wavefunction file (WFK.nc).

The class mirrors the interface of the `Wavecar` class so it can serve as a
drop-in replacement inside easyunfold.

Abinit WFK NetCDF layout
------------------------
Dimensions
    number_of_spins              (nspin)
    number_of_kpoints            (nkpt)
    max_number_of_coefficients   (mpw)   – padded to the largest k-point
    max_number_of_states          (mband)  - actual NetCDF dimension name in Abinit WFK files
    number_of_spinor_components  (nspinor) – 1 (collinear) or 2 (SOC)

Variables  [units]
    primitive_vectors              (3, 3)                        [Bohr]
    reduced_coordinates_of_kpoints (nkpt, 3)                    [reduced]
    number_of_coefficients         (nkpt,)                      [count]
    eigenvalues                    (nspin, nkpt, mband)          [Hartree]
    occupations                    (nspin, nkpt, mband)
    reduced_coordinates_of_plane_waves
                                   (nkpt, mpw, 3)               [reduced int]
    coefficients_of_wavefunctions  (nspin, nkpt, mband, nspinor, mpw) [complex]
    kinetic_energy_cutoff                                        [Hartree]
"""

import numpy as np

# ---------------------------------------------------------------------------
# Unit-conversion constants (kept consistent with vasp_constant.py values)
# ---------------------------------------------------------------------------
HARTREE_TO_EV = 27.211386245988   # 1 Ha → eV
BOHR_TO_ANG   = 0.529177210903    # 1 Bohr → Å
TPI           = 2.0 * np.pi


class AbinitWavefunction:  # pylint: disable=too-many-instance-attributes
    """
    Reader for Abinit pseudowavefunctions stored in a NetCDF WFK file.

    The public attributes and method signatures deliberately mirror those of
    the :class:`Wavecar` class so that band-unfolding code that works with
    ``Wavecar`` objects can accept an ``AbinitWavefunction`` with no further
    modifications.

    Parameters
    ----------
    fnm : str
        Path to the ``WFK.nc`` file.
    lsorbit : bool
        ``True`` if the file comes from a spin–orbit-coupling calculation
        (``nspinor == 2``).  When ``False`` the value is auto-detected from
        the file.
    """

    def __init__(self, fnm: str = 'WFK.nc', lsorbit: bool = False):
        try:
            import netCDF4 as nc  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "The 'netCDF4' package is required to read Abinit WFK files.\n"
                "Install it with:  pip install netCDF4"
            ) from exc

        self._fname = fnm
        self._ds    = nc.Dataset(fnm, 'r')

        # Read everything and close the file handle immediately so the object
        # is self-contained (mirrors Wavecar which keeps the file open but
        # this is safer for a NetCDF dataset).
        try:
            self._read_header()
            self._read_bands()
            self._read_gvectors()
            self._read_coefficients()
        finally:
            self._ds.close()
            self._ds = None

        # Validate SOC flag against what the file actually contains.
        detected_soc = (self._nspinor == 2)
        if lsorbit and not detected_soc:
            raise ValueError(
                "'lsorbit=True' was requested but the WFK file has "
                f"nspinor={self._nspinor}."
            )
        self._lsoc = detected_soc

        # Gamma-only WFK files are not produced by Abinit; flag is always False.
        self._lgam = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_header(self):
        """Read lattice, k-points, and scalar metadata."""
        ds = self._ds

        # ---- lattice (Bohr → Å) ----------------------------------------
        # Abinit stores rows as lattice vectors: shape (3, 3)
        pv = np.array(ds.variables['primitive_vectors'][:])   # Bohr
        self._realspace_cell          = pv * BOHR_TO_ANG       # Å
        self._realspace_cell_volume   = np.linalg.det(self._realspace_cell)
        # Reciprocal-cell matrix (rows = reciprocal vectors, no 2π factor —
        # matches Wavecar convention where TPI is applied separately)
        self._reciprocal_cell_volume  = np.linalg.inv(self._realspace_cell).T

        # ---- dimensions ------------------------------------------------
        self._nspin   = int(ds.dimensions['number_of_spins'].size)
        self._nkpts   = int(ds.dimensions['number_of_kpoints'].size)
        self._nbands  = int(ds.dimensions['max_number_of_states'].size)
        self._nspinor = int(ds.dimensions['number_of_spinor_components'].size)

        # ---- energy cut-off (Hartree → eV) ----------------------------
        self._encut = float(np.asarray(ds.variables['kinetic_energy_cutoff']).flat[0]) * HARTREE_TO_EV



        # ---- k-vectors (already in reduced / fractional coordinates) ---
        self._kvecs = np.array(ds.variables['reduced_coordinates_of_kpoints'][:])

        # ---- FFT grid dimensions (stored now while _ds is still open) ----
        # These are exposed via the mesh_size property in the wrapper class.
        _grid_keys = [
            'number_of_grid_points_vector1',
            'number_of_grid_points_vector2',
            'number_of_grid_points_vector3',
        ]
        if all(k in ds.dimensions for k in _grid_keys):
            self._ngrid = np.array(
                [int(ds.dimensions[k].size) for k in _grid_keys], dtype=int
            )
        else:
            self._ngrid = None

    def _read_bands(self):
        """Load eigenvalues (→ eV) and occupations; compute k-path length."""
        ds = self._ds

        # shape: (nspin, nkpt, mband) — Hartree → eV
        self._bands = np.array(ds.variables['eigenvalues'][:]) * HARTREE_TO_EV
        self._occs  = np.array(ds.variables['occupations'][:])

        # Number of plane waves actually used at each k-point
        self._nplws = np.array(ds.variables['number_of_coefficients'][:], dtype=int)

        # k-path arc length (same logic as Wavecar.read_bands)
        if self._nkpts > 1:
            tmp = np.linalg.norm(
                np.dot(
                    np.diff(self._kvecs, axis=0),
                    self._reciprocal_cell_volume
                ),
                axis=1
            )
            self._kpath = np.concatenate(([0], np.cumsum(tmp)))
        else:
            self._kpath = None     
        # Read Fermi energy directly from WFK (Hartree -> eV)
        if 'fermi_energy' in self._ds.variables:
            self._fermie = float(np.asarray(self._ds.variables['fermi_energy']).flat[0]) * HARTREE_TO_EV
        else:
            self._fermie = None
            
    def _read_gvectors(self):
        """
        Load or reconstruct G-vectors for every k-point.

        Abinit writes integer reduced coordinates into
        ``reduced_coordinates_of_plane_waves`` (shape nkpt × mpw × 3), but
        in many versions / run modes this array is left at the NetCDF fill
        value (INT32_MIN = -2147483648).  When that happens we fall back to
        reconstructing the G-sphere from scratch, exactly as VASP's Wavecar
        does in ``get_gvectors``, but using Hartree / Bohr atomic units.

        Reconstruction algorithm (mirrors wavecar.py get_gvectors):
        -----------------------------------------------------------
        1.  Build an FFT grid large enough to contain every G with
            KE = 0.5·|2π·(G+k)·B|² ≤ ecut  (Hartree, B = inv(rprim).T).
        2.  Wrap negative indices the same way as VASP:
              fi[ngrid//2 + 1 :] -= ngrid   (so range is -N/2 … N/2)
        3.  Form the full Cartesian product (gz slowest, gx fastest to match
            VASP convention — order does not matter because we sort by KE).
        4.  Filter by KE ≤ ecut and sort ascending in KE so the ordering is
            deterministic and matches Abinit's internal plane-wave ordering.
        5.  Verify the reconstructed count equals nplws[ik] from the file.
        """
        ds = self._ds

        # ── Try to use the stored G-vectors first ──────────────────────────
        SENTINEL = np.iinfo(np.int32).min          # -2147483648
        raw = np.array(
            ds.variables['reduced_coordinates_of_plane_waves'][:],
            dtype=np.int32,
        )
        if not np.any(raw == SENTINEL):
            # File contains valid G-vectors — use them directly.
            self._all_gvectors = [
                raw[ik, : self._nplws[ik], :].astype(int)
                for ik in range(self._nkpts)
            ]
            return

        # ── Fallback: reconstruct from ecut (Hartree / Bohr) ───────────────
        # ecut is stored in Hartree; _encut was already converted to eV,
        # so read the raw value again from the still-open dataset.
        ecut_ha = float(
            np.asarray(ds.variables['kinetic_energy_cutoff']).flat[0]
        )                                           # Hartree

        # rprim rows are lattice vectors in Bohr.
        rprim = np.array(ds.variables['primitive_vectors'][:])   # (3,3) Bohr

        # Reciprocal-cell matrix: rows = reciprocal lattice vectors (Bohr⁻¹),
        # without the 2π factor — TPI is applied in the KE expression below.
        B = np.linalg.inv(rprim).T                 # (3,3)

        # Minimum reciprocal-vector length → upper bound on |G| components.
        b_norms  = np.linalg.norm(TPI * B, axis=1)          # |b_i| in Bohr⁻¹
        b_min    = b_norms.min()
        # Maximum |G| from KE ≤ ecut:  0.5*(|G|*b_min)² ≤ ecut
        gmax = int(np.ceil(np.sqrt(2.0 * ecut_ha) / b_min)) + 1

        # Build the FFT index ranges with negative-index wrap (VASP convention)
        ngrid = 2 * gmax + 1
        fi    = np.arange(ngrid, dtype=int)
        fi[ngrid // 2 + 1:] -= ngrid               # range: -gmax … +gmax

        # Cartesian product: gz slowest, gx fastest (matches VASP meshgrid)
        gz, gy, gx = np.meshgrid(fi, fi, fi, indexing='ij')
        gvecs_all  = np.stack(
            [gx.ravel(), gy.ravel(), gz.ravel()], axis=1
        )                                           # (N, 3)

        self._all_gvectors = []
        for ik in range(self._nkpts):
            kvec = self._kvecs[ik]                  # fractional

            # KE = 0.5 · |2π · (G + k) · B|²  in Hartree (atomic units ħ=m=1)
            kg_cart = TPI * np.dot(gvecs_all + kvec, B)   # (N, 3) Bohr⁻¹
            KE      = 0.5 * np.sum(kg_cart ** 2, axis=1)  # (N,)   Hartree

            inside  = np.where(KE <= ecut_ha)[0]
            # Sort by ascending KE so ordering is deterministic
            order   = inside[np.argsort(KE[inside], kind='stable')]
            gvecs_k = gvecs_all[order].astype(int)  # (nplw, 3)

            # Sanity check against the count stored in the file
            nplw_file = self._nplws[ik]
            if gvecs_k.shape[0] != nplw_file:
                raise ValueError(
                    f"G-vector count mismatch at k-point {ik + 1}: "
                    f"reconstructed {gvecs_k.shape[0]}, "
                    f"file says {nplw_file}.  "
                    "Check ecut and istwfk settings."
                )

            self._all_gvectors.append(gvecs_k)
        
    def _read_coefficients(self):
        """
        Load all plane-wave coefficients into memory.

        Shape in file: ``(nspin, nkpt, mband, nspinor, max_nplw)``.
        The array is complex128 after conversion.

        Padding columns (beyond ``_nplws[ik]``) are stripped per k-point and
        stored as a nested structure::

            _coeffs[ispin][ikpt][iband]  →  np.ndarray, shape (nspinor, nplw)
        """
        ds = self._ds
        # netCDF4-python returns a MaskedArray; fill_value is the NetCDF
        # default 9.96920997e+36.  Calling np.array() on a MaskedArray
        # replaces masked elements with the fill value, which makes it
        # impossible to distinguish "genuinely zero" from "not written".
        # Use np.ma.filled(..., 0) instead so masked → 0+0j and real data
        # is preserved, then detect whether Abinit actually wrote anything.
        raw_ma = ds.variables['coefficients_of_wavefunctions'][:]  # MaskedArray

        # Detect if ALL values are masked (coefficients not written by Abinit)
        if np.ma.is_masked(raw_ma) and raw_ma.mask.all():
            raise ValueError(
                "coefficients_of_wavefunctions is entirely masked in "
                f"'{self._fname}'.\n"
                "Abinit did not write the wavefunction coefficients.\n"
                "Add 'prtwf 1' (or 'prtwf2 1' for dataset 2) to your "
                "Abinit input and rerun."
            )

        # Replace masked entries (padding / not-written) with 0
        raw_data = np.ma.filled(raw_ma, fill_value=0.0)

        # Last dimension is 2 (real, imag) for NetCDF complex storage
        if raw_data.shape[-1] == 2 and not np.iscomplexobj(raw_data):
            raw = raw_data[..., 0] + 1j * raw_data[..., 1]
        else:
            raw = raw_data.astype(np.complex128)
        # raw shape: (nspin, nkpt, mband, nspinor, max_nplw)

        self._coeffs = [
            [
                [
                    raw[ispin, ik, ib, :, : self._nplws[ik]]
                    for ib in range(self._nbands)
                ]
                for ik in range(self._nkpts)
            ]
            for ispin in range(self._nspin)
        ]
        
    # ------------------------------------------------------------------
    # Public interface — mirrors Wavecar
    # ------------------------------------------------------------------

    def is_soc(self) -> bool:
        """Return ``True`` if the WFK comes from a spin–orbit calculation."""
        return bool(self._lsoc)

    def is_gamma(self) -> bool:
        """
        Always ``False``: Abinit does not produce gamma-only WFK files in the
        same sense as VASP.
        """
        return False

    def get_gvectors(self, ikpt: int = 1, **_kwargs) -> np.ndarray:
        """
        Return the G-vectors for k-point *ikpt* (1-based) as an integer array
        of shape ``(nplw, 3)`` in reduced (fractional) coordinates.

        Unlike the VASP version, G-vectors are read directly from the file
        rather than being reconstructed from the energy cut-off, so
        ``force_gamma`` and ``check_consistency`` keyword arguments are
        accepted but silently ignored.

        Parameters
        ----------
        ikpt : int
            1-based k-point index.
        """
        assert 1 <= ikpt <= self._nkpts, (
            f"Invalid k-point index {ikpt}; valid range 1–{self._nkpts}."
        )
        return self._all_gvectors[ikpt - 1].copy()

 
    def read_band_coeffs(
        self,
        ispin: int = 1,
        ikpt:  int = 1,
        iband: int = 1,
        norm:  bool = False,
    ) -> np.ndarray:
        """
        Return the plane-wave coefficients for the requested KS state.

        For a collinear calculation the returned array has shape ``(nplw,)``.
        For an SOC calculation it has shape ``(2 * nplw,)`` with the two
        spinor components concatenated, which matches the convention used by
        the VASP ``Wavecar`` class.

        Parameters
        ----------
        ispin : int
            1-based spin index (1 or 2 for collinear; always 1 for SOC).
        ikpt  : int
            1-based k-point index.
        iband : int
            1-based band index.
        norm  : bool
            If ``True`` the coefficients are normalised to unit norm before
            returning.
        """
        self._check_index(ispin, ikpt, iband)

        # _coeffs[ispin-1][ikpt-1][iband-1] shape: (nspinor, nplw)
        cg = self._coeffs[ispin - 1][ikpt - 1][iband - 1].copy()

        if self._lsoc:
            # Flatten spinor components: (2, nplw) → (2*nplw,)
            cg = cg.reshape(-1)
        else:
            # Collinear: nspinor == 1, drop that axis → (nplw,)
            cg = cg[0]

        if norm:
            cg /= np.linalg.norm(cg)

        return cg 

    def _check_index(self, ispin: int, ikpt: int, iband: int):
        """Validate spin / k-point / band indices (1-based)."""
        assert 1 <= ispin <= self._nspin,  f"Invalid spin index {ispin}!"
        assert 1 <= ikpt  <= self._nkpts,  f"Invalid k-point index {ikpt}!"
        assert 1 <= iband <= self._nbands, f"Invalid band index {iband}!"    
#--------------------------Test---------------------------------#
"""
if __name__ == "__main__":
    import sys
    t1 = AbinitWavefunction(fnm = 'trf2_1o_DS2_WFK.nc')
    print("ngrid:", t1._encut)
"""