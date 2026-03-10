"""
Compatibility layers for handling wavefunctions
"""

# pylint: disable= protected-access, useless-super-delegation
import numpy as np
from ase.units import Hartree

from castepxbin.wave import WaveFunction as CastepWF
from .wavecar import Wavecar
from .abinit_wfk import AbinitWavefunction as WFKReader


class WaveFunction:
    """
    Interface for accessing wavefunction data

    All indexings are one-base rather than zero-based as in python
    """

    def __init__(self, wfc):
        self.wfc = wfc

    @property
    def kpoints(self):
        """Kpoints as row vectors"""
        raise NotImplementedError

    @property
    def nkpts(self):
        return self.kpoints.shape[0]

    @property
    def nspins(self):
        """Number of spins"""
        raise NotImplementedError

    @property
    def mesh_size(self):
        raise NotImplementedError

    @property
    def bands(self):
        """KS band energies in shape (ns, nk, nb)"""
        raise NotImplementedError

    @property
    def nbands(self):
        """Number of KS bands"""
        return self.bands.shape[-1]

    @property
    def occupancies(self):
        """Occupancies of each band"""
        raise NotImplementedError

    def get_gvectors(self, ik):
        """Return the gvectors at a kpoint with shape (nwaves, 3)"""
        raise NotImplementedError

    def get_band_coeffs(self, ispin, ik, ib, norm=True):
        """Return band coefficients"""
        raise NotImplementedError


class VaspWaveFunction(WaveFunction):
    """Interface for accessing WAVECAR"""

    def __init__(self, wfc: Wavecar):
        super().__init__(wfc)

    @property
    def kpoints(self):
        return self.wfc._kvecs

    @property
    def occupancies(self):
        return self.wfc._occs

    @property
    def nspins(self):
        return self.wfc._nspin

    @property
    def mesh_size(self):
        return self.wfc._ngrid

    @property
    def bands(self):
        """KS band energies in shape (ns, nk, nb)"""
        return self.wfc._bands

    def get_gvectors(self, ik):
        """Return the gvectors at a kpoint"""
        return self.wfc.get_gvectors(ik)

    def get_band_coeffs(self, ispin, ik, ib, norm=True):
        """Return plane wave coefficients at a specific band"""
        return self.wfc.read_band_coeffs(ispin, ik, ib, norm=norm)


class CastepWaveFunction(WaveFunction):
    """
    Interface for reading wave function data from a CASTEP calculation.
    """

    def __init__(self, wfc: CastepWF):
        super().__init__(wfc)
        self.wfc: CastepWF

    @classmethod
    def from_file(cls, fname):
        return cls(CastepWF.from_file(fname))

    @property
    def kpoints(self):
        return self.wfc.kpts.T

    @property
    def occupancies(self):
        return self.wfc.occupancies.T

    @property
    def mesh_size(self):
        return self.wfc.mesh_size

    @property
    def bands(self):
        return self.wfc.eigenvalues.T * Hartree

    @property
    def nspins(self):
        return self.wfc.nspins

    def get_gvectors(self, ik):
        """Return the G-vector at a kpoint"""
        return self.wfc.get_gvectors(ik - 1).T

    def get_band_coeffs(self, ispin, ik, ib, norm=True):
        """Return the plane wave coefficients for a band"""
        coeffs = self.wfc.get_plane_wave_coeffs(ispin - 1, ik - 1, ib - 1)
        if norm:
            coeffs = coeffs / np.linalg.norm(coeffs)
        return coeffs


class AbinitWaveFunction(WaveFunction):
    """
    Interface for reading wavefunction data from an ABINIT calculation.

    Wraps the :class:`~.abinit_wfk.AbinitWavefunction` reader so that ABINIT
    WFK files can be used interchangeably with :class:`VaspWaveFunction` and
    :class:`CastepWaveFunction` in band-unfolding workflows.

    Notes
    -----
    * All k-point, spin, and band indices passed to methods are **1-based**,
      consistent with the rest of this module.
    * Eigenvalues (``bands``) are returned in **eV**.  The underlying
      :class:`AbinitWavefunction` reader always converts from Hartree on load.
    * ``mesh_size`` returns the real-space FFT grid ``[ng1, ng2, ng3]`` when
      the ``number_of_grid_points_vector*`` dimensions are present in the file,
      or ``None`` otherwise.
    * For SOC calculations (``nspinor == 2``) ``get_band_coeffs`` returns a
      ``(2*nplw,)`` array with spin-up and spin-down components concatenated,
      matching the convention of :class:`VaspWaveFunction`.
    """

    def __init__(self, wfc: WFKReader):
        super().__init__(wfc)
        self.wfc: WFKReader

    @classmethod
    def from_file(cls, fname: str, lsorbit: bool = False):
        """
        Construct an :class:`AbinitWaveFunction` directly from a WFK file path.

        Parameters
        ----------
        fname : str
            Path to the ``*_WFK.nc`` file.
        lsorbit : bool
            Set ``True`` for spin–orbit coupling (noncollinear) calculations.
            The flag is also auto-detected from ``nspinor == 2`` in the file.

        Returns
        -------
        AbinitWaveFunction
        """
        return cls(WFKReader(fnm=fname, lsorbit=lsorbit))

    # ------------------------------------------------------------------
    # Core interface properties
    # ------------------------------------------------------------------

    @property
    def kpoints(self):
        """K-points as row vectors in reduced coordinates, shape (nkpt, 3)."""
        return self.wfc._kvecs

    @property
    def nspins(self):
        """Number of spin channels (1 for unpolarised/SOC, 2 for spin-polarised)."""
        return self.wfc._nspin

    @property
    def mesh_size(self):
        """
        Real-space FFT grid dimensions ``[ng1, ng2, ng3]``.

        Populated from ``number_of_grid_points_vector1/2/3`` dimensions during
        file read.  Returns ``None`` if those dimensions were absent.
        """
        return self.wfc._ngrid  # captured at read time; _ds is already closed

    @property
    def bands(self):
        """KS eigenvalues in eV, shape (nspin, nkpt, nbands)."""
        return self.wfc._bands

    @property
    def occupancies(self):
        """Band occupancies, shape (nspin, nkpt, nbands)."""
        return self.wfc._occs
    
    @property
    def fermie(self):
        """Fermi energy in eV, read directly from WFK file."""
        return self.wfc._fermie

    # ------------------------------------------------------------------
    # G-vectors and coefficients
    # ------------------------------------------------------------------

    def get_gvectors(self, ik):
        """
        Return the G-vectors at a k-point as integer Miller indices.

        Parameters
        ----------
        ik : int
            1-based k-point index.

        Returns
        -------
        gvecs : ndarray, shape (nplw, 3), dtype int
            G-vectors in reduced (fractional reciprocal) coordinates.
        """
        return self.wfc.get_gvectors(ik)

    def get_band_coeffs(self, ispin, ik, ib, norm=True):
        """
        Return the plane-wave coefficients for a specified KS state.

        Parameters
        ----------
        ispin : int   1-based spin index
        ik    : int   1-based k-point index
        ib    : int   1-based band index
        norm  : bool  If ``True`` (default), normalise so that ``∑|c|² = 1``

        Returns
        -------
        coeffs : ndarray, complex128
            Shape ``(nplw,)`` for collinear calculations, or ``(2*nplw,)``
            for SOC (spin-up and spin-down concatenated).
        """
        return self.wfc.read_band_coeffs(ispin, ik, ib, norm=norm)