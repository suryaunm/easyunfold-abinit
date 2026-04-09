"""
Overlay multiple easyunfold band structures on one plot.

Fully STANDALONE — only requires numpy, matplotlib, and the standard
library.  No easyunfold installation needed.

USAGE
-----
1. Edit the USER SETTINGS section (file paths, labels, colours, eref).
2. Run in Spyder with F5, or from the command line:  python plot_multi_unfold.py

REQUIREMENTS  (install once)
-----------------------------
    pip install numpy matplotlib
"""

import json
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

# ============================================================
#  USER SETTINGS — edit everything in this block
# ============================================================

# Full paths to the easyunfold.json files.
# On Windows use raw strings  r"C:\path\to\file"  or forward slashes.
JSON_FILES = [
    "easyunfold_Mo.json",
    "easyunfold_Nb.json",
    "easyunfold_Ta.json",
    "easyunfold_W.json",
    # add more entries as needed:
    # r"C:\Users\YourName\calc\run3\easyunfold.json",
]

# Legend label for each file (same order as JSON_FILES).
LABELS = [
    "Mo",
    "Nb",
    "Ta",
    "W",
    # "Run 3",
]

# Colour for each dataset — any matplotlib colour string or hex code.
# Good two-dataset pairs:
#   blue + red    : "#1f77b4", "#d62728"
#   teal + orange : "#17becf", "#ff7f0e"
#   green + purple: "#2ca02c", "#9467bd"
COLORS = [
    "#1f77b4",   # blue
    "#d62728",   # red
    "#2ca02c", # green  (uncomment for a third dataset)
    "#9467bd", # purple (uncomment for a fourth dataset)
]

# Fermi energy (eV) used as the zero of the energy axis.
# Use a single float to apply the same value to all files:
#   EREF = 17.02
# Or a list with one value per file for different calculations:
EREF = [17.038114, 14.3471, 13.3473, 19.1785]
#EREF = 17.02

# Energy window [emin, emax] in eV relative to EREF shown on the plot.
YLIM = (-5, 9)

# Specific band indices (0-indexed) to plot. Set to None to plot all bands.
# Use a single list like [10, 11, 12] or list(range(10, 20)) for all files,
# or a list of lists for per-file bands.
BANDS_TO_PLOT = [10]

# ── Spectral function settings ────────────────────────────────────────────────
NPOINTS = 2000   # number of energy grid points for the spectral function
SIGMA   = 0.02   # REDUCED FROM 0.2: Gaussian broadening in eV (smaller = sharper, less diffused)

# ── Appearance ────────────────────────────────────────────────────────────────
FIGSIZE    = (5, 4)       # (width, height) in inches
DPI        = 200          # figure resolution in dots per inch
INTENSITY  = 5.0          # higher = brighter bands  (try 3–10)
VSCALE     = 1.0          # colour-scale normalisation (lower = more intense)
ALPHA      = 0.75         # per-dataset transparency  (0 = invisible, 1 = solid)
ZERO_LINE  = True         # draw a dashed horizontal line at E = 0 (Fermi level)
LEGEND_LOC = "upper left"

# ── Output ────────────────────────────────────────────────────────────────────
# Set to None to skip saving, or provide a file path.
SAVE = None
# SAVE = r"C:\Users\YourName\Desktop\comparison.png"

# ============================================================
#  END OF USER SETTINGS
# ============================================================


# ── JSON loading helpers ──────────────────────────────────────────────────────

def _decode_numpy(obj):
    """
    Recursively decode objects serialised by monty / easyunfold.
    These have the form:
        {"@module": "numpy", "@class": "array", "dtype": "...", "data": [...]}
    and are returned as numpy arrays.  All other dicts and lists are
    passed through unchanged.
    """
    if isinstance(obj, dict):
        if obj.get("@module") == "numpy" and obj.get("@class") == "array":
            return np.array(obj["data"], dtype=obj.get("dtype", "float64"))
        return {k: _decode_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_numpy(v) for v in obj]
    return obj


def load_json(path):
    """
    Load an easyunfold.json file and return a plain Python dict with all
    numpy arrays already decoded.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return _decode_numpy(raw)


# ── k-point distance calculation ─────────────────────────────────────────────

def get_kpoint_distances(data, hide_discontinuities=True):
    """
    Compute cumulative k-path distances in Angstrom^-1.

    Replicates UnfoldKSet.get_kpoint_distances() without importing easyunfold.

    Parameters
    ----------
    data : dict
        Decoded easyunfold.json dictionary.
    hide_discontinuities : bool
        When True, collapse the gap at path discontinuities so the segments
        are joined on the plot (same as easyunfold default behaviour).

    Returns
    -------
    dists : np.ndarray, shape (nkpts,)
        Cumulative distances along the k-path.
    """
    kpts_pc = np.array(data["kpts_pc"])    # (nkpts, 3) fractional coords
    pc_latt = np.array(data["pc_latt"])    # (3, 3) lattice vectors in Angstrom

    # Reciprocal lattice with 2pi factor: B = 2pi * (L^{-1})^T
    recip = np.linalg.inv(pc_latt).T * 2.0 * np.pi   # (3, 3)

    # Convert fractional k-path to Cartesian Angstrom^-1
    kpath_cart = kpts_pc @ recip            # (nkpts, 3)

    dists = np.cumsum(np.linalg.norm(np.diff(kpath_cart, axis=0), axis=-1))
    dists = np.append([0.0], dists)

    if hide_discontinuities:
        kpoint_labels = data.get("kpoint_labels", [])
        last_idx = -2
        for idx, _ in kpoint_labels:
            if idx - last_idx == 1:
                shift = dists[idx] - dists[idx - 1]
                dists[idx:] -= shift
            last_idx = idx

    return dists


def get_combined_kpoint_labels(data):
    """
    Return high-symmetry labels with adjacent discontinuities merged
    into a single label separated by '|'.

    Replicates UnfoldKSet.get_combined_kpoint_labels().
    """
    kpoint_labels = data.get("kpoint_labels", [])
    last_entry    = [-2, None]
    combined      = []
    for idx, name in kpoint_labels:
        if idx - last_entry[0] == 1:
            combined.append([last_entry[0], last_entry[1] + "|" + name])
        else:
            combined.append([idx, name])
        last_entry = [idx, name]
    return combined


# ── spectral function smearing ────────────────────────────────────────────────

def _lorentzian(x, x0, sigma):
    """Normalised Lorentzian centred on x0 (matches easyunfold default)."""
    return (1.0 / np.pi) * sigma**2 / ((x - x0)**2 + sigma**2)


def compute_spectral_function(data, npoints=2000, sigma=0.1, bands=None):
    """
    Reconstruct the spectral function A(k, E) from pre-computed spectral
    weights stored in easyunfold.json.

    Replicates spectral_function_from_weight_sets() from unfold.py.

    The spectral weights live in:
        calculated_quantities["spectral_weights_per_set"]
    which is a list of length nkpts_pc.  Each element is an array of shape
        (nspin, n_expanded_kpts, nbands, 2)
    where the last axis stores [energy_eV, spectral_weight].

    Parameters
    ----------
    data : dict
        Decoded easyunfold.json dictionary.
    npoints : int
        Number of energy grid points.
    sigma : float
        Lorentzian broadening width in eV.

    Returns
    -------
    eng : np.ndarray, shape (npoints,)
        Energy grid in eV.
    sf  : np.ndarray, shape (nspin, npoints, nkpts_pc)
        Spectral function.
    """
    sws_raw      = data["calculated_quantities"]["spectral_weights_per_set"]
    kweight_sets = data["expansion_results"]["weights"]

    nk    = len(sws_raw)
    nspin = np.array(sws_raw[0]).shape[0]

    # Global energy range across all k-points and bands
    if bands is not None:
        e_all = np.concatenate([np.array(sw)[:, :, bands, 0].ravel() for sw in sws_raw])
    else:
        e_all = np.concatenate([np.array(sw)[:, :, :, 0].ravel() for sw in sws_raw])
    emin, emax = float(e_all.min()), float(e_all.max())
    eng = np.linspace(emin - 5.0 * sigma, emax + 5.0 * sigma, npoints)

    sf = np.zeros((nspin, npoints, nk), dtype=float)

    for ispin, ik in itertools.product(range(nspin), range(nk)):
        sw_set = np.array(sws_raw[ik])      # (nspin, n_sub, nbands, 2)
        kws    = kweight_sets[ik]           # weights for each sub-k-point

        n_sub = sw_set.shape[1]
        for jsub in range(n_sub):
            # Extract the scalar weight for this sub-k-point
            raw_kw = kws[jsub]
            kw = float(np.array(raw_kw).ravel()[0])

            if bands is not None:
                E_Km = sw_set[ispin, jsub, bands, 0]   # (selected bands,)
                P_Km = sw_set[ispin, jsub, bands, 1]   # (selected bands,)
            else:
                E_Km = sw_set[ispin, jsub, :, 0]   # (nbands,) energies
                P_Km = sw_set[ispin, jsub, :, 1]   # (nbands,) spectral weights

            # Lorentzian smearing: shape (npoints, nbands) -> sum over bands
            contrib = _lorentzian(
                eng[:, np.newaxis],    # (npoints, 1)
                E_Km[np.newaxis, :],   # (1, nbands)
                sigma,
            )
            sf[ispin, :, ik] += (contrib * P_Km[np.newaxis, :]).sum(axis=1) * kw

    return eng, sf


# ── k-point label drawing ─────────────────────────────────────────────────────

def _clean_label(label):
    """Convert high-symmetry labels to pretty matplotlib math strings."""
    label = label.strip()

    # Handle pipe-joined discontinuity labels like "H|P"
    if "|" in label:
        parts = label.split("|")
        return "|".join(_clean_label(p) for p in parts)

    special = {
        r"\Gamma": r"$\Gamma$",
        "Gamma":   r"$\Gamma$",
        "GAMMA":   r"$\Gamma$",
        "G":       r"$\Gamma$",
    }
    if label in special:
        return special[label]

    # Wrap in upright math font
    return r"$\mathrm{" + label + r"}$"


def add_kpoint_labels(ax, data, kdist):
    """Draw vertical dashed lines and tick labels at high-symmetry k-points."""
    combined = get_combined_kpoint_labels(data)
    tick_locs, tick_labels = [], []
    for idx, label in combined:
        xloc = float(kdist[idx])
        ax.axvline(x=xloc, lw=0.5, color="k", ls=":", alpha=0.8)
        tick_locs.append(xloc)
        tick_labels.append(_clean_label(label))
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)


# ── main plotting function ────────────────────────────────────────────────────

def plot_multi_unfold(
    json_files,
    labels,
    colors,
    eref,
    ylim=(-5, 9),
    bands_to_plot=None,
    npoints=2000,
    sigma=0.02,
    figsize=(4, 3),
    dpi=300,
    intensity=1.0,
    vscale=1.0,
    alpha=0.75,
    zero_line=True,
    legend_loc="upper right",
    save=None,
):
    """
    Load multiple easyunfold.json files and overlay their unfolded band
    structures on a single plot, each drawn in a distinct colour with a legend.

    Parameters
    ----------
    json_files : list of str
        Paths to easyunfold.json files.  All files must share the same k-path.
    labels : list of str
        Legend label for each dataset.
    colors : list of str
        Matplotlib colour for each dataset (hex, named colour, etc.).
    eref : float or list of float
        Fermi energy in eV.  A single float applies to all datasets.
    ylim : (float, float)
        Energy window [emin, emax] in eV relative to eref.
    npoints : int
        Energy grid points for the spectral function.
    sigma : float
        Lorentzian broadening width in eV.
    figsize : (float, float)
        Figure (width, height) in inches.
    dpi : float
        Figure resolution in dots per inch.
    intensity : float
        Colour intensity multiplier — higher = brighter bands (try 3–10).
    vscale : float
        Colour-scale normalisation factor (lower = more intense).
    alpha : float
        Per-dataset transparency (0 = invisible, 1 = fully opaque).
    zero_line : bool
        Draw a dashed horizontal line at E = 0 (the Fermi level).
    legend_loc : str
        Legend position string accepted by ax.legend().
    save : str or None
        File path to save the figure.  None = do not save.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(json_files)

    # ── input validation ──────────────────────────────────────────────────────
    if len(labels) != n:
        raise ValueError(
            f"len(labels)={len(labels)} does not match "
            f"len(json_files)={n}."
        )
    if len(colors) != n:
        raise ValueError(
            f"len(colors)={len(colors)} does not match "
            f"len(json_files)={n}."
        )
    erefs = ([float(eref)] * n
             if isinstance(eref, (int, float))
             else list(float(e) for e in eref))
    if len(erefs) != n:
        raise ValueError(
            f"len(eref)={len(erefs)} does not match "
            f"len(json_files)={n}."
        )

    # Normalize bands_to_plot list
    if bands_to_plot is None:
        b_list = [None] * n
    elif all(isinstance(b, int) for b in bands_to_plot) and len(bands_to_plot) > 0:
        b_list = [bands_to_plot] * n
    elif len(bands_to_plot) == n:
        b_list = bands_to_plot
    else:
        raise ValueError("BANDS_TO_PLOT must be None, a list of ints, or a list of lists matching the number of files.")

    # ── create figure ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    legend_patches = []
    kdist_ref      = None   # k-distances from first file; shared x-axis

    for i, (path, label, color, this_eref, these_bands) in enumerate(
            zip(json_files, labels, colors, erefs, b_list)):

        print(f"[{i + 1}/{n}] Loading    : {path}")

        # ── load and validate ─────────────────────────────────────────────────
        try:
            data = load_json(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File not found: {path}\n"
                "Check the path in the JSON_FILES list at the top of the script."
            )

        calc = data.get("calculated_quantities", {})
        if "spectral_weights_per_set" not in calc:
            raise RuntimeError(
                f"\nNo spectral weights found in:\n  {path}\n\n"
                "You need to run the unfold calculation first:\n"
                "  easyunfold unfold calculate <WFK_or_WAVECAR_file>\n"
                "This creates / updates easyunfold.json with the spectral weights."
            )

        # ── k-point distances ─────────────────────────────────────────────────
        kdist = get_kpoint_distances(data)
        if kdist_ref is None:
            kdist_ref = kdist

        # ── compute spectral function ─────────────────────────────────────────
        print(f"           Computing spectral function  "
              f"(npoints={npoints}, sigma={sigma} eV) ...")
        eng, sf = compute_spectral_function(data, npoints=npoints, sigma=sigma, bands=these_bands)

        nspin = sf.shape[0]
        nengs = sf.shape[1]
        ebin  = (eng.max() - eng.min()) / max(nengs - 1, 1)

        # ── colour-scale normalisation ────────────────────────────────────────
        mask    = (eng < (ylim[1] + this_eref)) & (eng > (ylim[0] + this_eref))
        sf_view = sf[:, mask, :]
        vmin_v  = float(sf_view.min())
        vmax_v  = float(sf_view.max())
        vmax_v  = (vmax_v - vmin_v) * (vscale / max(intensity, 1e-12)) + vmin_v

        # ── build RGBA image array ────────────────────────────────────────────
        # RGB channels = chosen colour (constant)
        # A channel    = normalised spectral weight × per-dataset alpha
        denom    = max(vmax_v - vmin_v, 1e-12)
        alpha_ch = np.clip((sf - vmin_v) / denom * alpha, 0.0, 1.0)

        rgb = np.array(to_rgb(color), dtype=np.float32)   # (3,)

        # shape (nspin, nengs, nk, 4)
        sf_rgba         = np.zeros(sf.shape + (4,), dtype=np.float32)
        sf_rgba[..., 0] = rgb[0]
        sf_rgba[..., 1] = rgb[1]
        sf_rgba[..., 2] = rgb[2]
        sf_rgba[..., 3] = alpha_ch.astype(np.float32)

        # ── draw each spin channel ────────────────────────────────────────────
        extent = [
            float(kdist.min()),
            float(kdist.max()),
            float(eng.min() - this_eref - ebin * 0.5),
            float(eng.max() - this_eref + ebin * 0.5),
        ]

        for ispin in range(nspin):
            ax.imshow(
                sf_rgba[ispin],
                extent=extent,
                aspect="auto",
                origin="lower",
                interpolation="bilinear",
            )

        legend_patches.append(
            Patch(facecolor=color, label=label, alpha=0.9)
        )

        # Draw k-point labels from the first dataset
        # (all datasets share the same k-path so labels are identical)
        if i == 0:
            add_kpoint_labels(ax, data, kdist)

        print(f"           Done  (eref = {this_eref:.4f} eV)")

    # ── final axes formatting ─────────────────────────────────────────────────
    ax.set_ylim(ylim)
    ax.set_xlim(float(kdist_ref.min()), float(kdist_ref.max()))
    ax.set_ylabel("Energy (eV)", labelpad=5)

    if zero_line:
        ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6, zorder=10)

    ax.legend(
        handles=legend_patches,
        loc=legend_loc,
        fontsize=9,
        frameon=True,
        framealpha=0.85,
        handlelength=1.2,
    )

    fig.tight_layout(pad=0.3)
    fig.savefig("BS_ele_bands.png", dpi=dpi, bbox_inches="tight")

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save}")

    return fig


# ============================================================
#  ENTRY POINT — runs when you press F5 in Spyder
# ============================================================
if __name__ == "__main__":

    # When running outside Spyder (e.g. plain terminal without a display)
    # switch to the non-interactive Agg backend so savefig still works.
    if not matplotlib.is_interactive():
        matplotlib.use("Agg")

    print("=" * 56)
    print("  easyunfold multi-JSON overlay plotter (standalone)")
    print("=" * 56)

    fig = plot_multi_unfold(
        json_files    = JSON_FILES,
        labels        = LABELS,
        colors        = COLORS,
        eref          = EREF,
        ylim          = YLIM,
        bands_to_plot = BANDS_TO_PLOT,
        npoints       = NPOINTS,
        sigma         = SIGMA,
        figsize       = FIGSIZE,
        intensity   = INTENSITY,
        vscale      = VSCALE,
        alpha       = ALPHA,
        zero_line   = ZERO_LINE,
        legend_loc  = LEGEND_LOC,
        save        = SAVE,
    )

    plt.show()
    print("Done.")
