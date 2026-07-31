"""Thermo-oxidative ageing indices from a pyrogram.

How degraded a recyclate is decides what it can be used for, and unlike
composition it cannot be read off a single peak. The indices here are ratios,
which is deliberate: a ratio is insensitive to how much sample was pyrolysed, to
detector gain, and to the injection split — none of which the analyst controls
tightly enough for absolute intensities to mean anything between laboratories.

Five quantities, each with a distinct mechanism behind it:

* **Carbonyl index** — oxidation products over saturated hydrocarbons. The
  primary evidence that oxidation happened at all.
* **Acid share of the oxidation products** — aldehydes are early intermediates,
  ketones accumulate, carboxylic acids are the end of the road. The share of
  acids therefore separates *mildly* from *severely* aged material, which the
  carbonyl index alone cannot.
* **Alkene/alkane ratio** — chain scission leaves terminal double bonds.
* **Branching index** — iso-alkenes over n-alkanes. Distinguishes LDPE from HDPE
  in virgin material and rises with ageing as secondary radical attack creates
  new branch points. It is therefore reported separately, not folded into the
  degradation score.
* **Mean chain length** — repeated processing shortens chains, moving the
  pyrolysate distribution to lower carbon numbers.

Absolute values are close to meaningless: every one of them depends on the
pyrolysis temperature and on the polymer. They are interpretable only against a
virgin reference of the same material, which is why
:meth:`DegradationIndices.relative_to` exists and why the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.matrix.polyolefin import PolyolefinMatrixModel

__all__ = [
    "OXIDATION_IONS",
    "HYDROCARBON_IONS",
    "DegradationIndices",
    "compute_degradation_indices",
    "chain_length_statistics",
]



# Diagnostic ions. These are fragmentation facts, not library identifications:
# the McLafferty rearrangement of an aldehyde gives m/z 44, of a methyl ketone
# m/z 58, of a carboxylic acid m/z 60 with its m/z 73 companion. They are quiet in
# a pure hydrocarbon spectrum, which is what makes them usable against a
# polyolefin background.
OXIDATION_IONS: dict[str, tuple[float, ...]] = {
    "aldehyde": (44.0,),
    "ketone": (58.0,),
    # m/z 60 only, deliberately. The carboxylic-acid McLafferty ion has m/z 73 as
    # its companion, and using both is the textbook pairing — but on a
    # polysiloxane column m/z 73 is also *the* column-bleed ion. Including it made
    # the measured acid share fall with ageing where the true share rises: at low
    # ageing the bleed dominated the channel, and as real oxidation grew the
    # ketone and aldehyde ions outpaced it. m/z 60 alone tracks the truth.
    "acid": (60.0,),
}

_ACID_SHARE_MIN_CARBONYL = 0.03
"""Carbonyl index below which the acid share is background-dominated."""

HYDROCARBON_IONS: dict[str, tuple[float, ...]] = {
    "alkane": (43.0, 57.0, 71.0, 85.0),
    "alkene": (41.0, 55.0, 69.0, 83.0),
    "branched": (56.0, 70.0, 84.0),
}


@dataclass(frozen=True, slots=True)
class DegradationIndices:
    """Ageing indicators of one sample.

    Attributes:
        carbonyl_index: Oxidation-product signal over alkane signal.
        acid_share: Acids as a fraction of all oxidation products, in ``[0, 1]``.
        alkene_to_alkane: Terminal-unsaturation indicator.
        branching_index: Iso-alkene over n-alkane signal.
        mean_carbon_number: Area-weighted mean chain length of the comb, or
            ``None`` when no comb model was supplied.
        chain_length_spread: Standard deviation of that distribution.
        basis: What the indices were computed from, for the report's provenance.
        notes: Caveats that must travel with the numbers.
    """

    carbonyl_index: float
    acid_share: float
    alkene_to_alkane: float
    branching_index: float
    mean_carbon_number: float | None
    chain_length_spread: float | None
    basis: str
    notes: tuple[str, ...] = ()

    def relative_to(self, reference: DegradationIndices) -> dict[str, float]:
        """Express these indices as multiples of a virgin reference.

        The only form in which they should be reported to a customer. An absolute
        carbonyl index of 0.02 means nothing on its own; three times the virgin
        value of the same grade means something.

        Args:
            reference: Indices of virgin material of the same polymer, measured
                under the same pyrolysis conditions.

        Returns:
            Ratios keyed by index name. A reference value of zero yields ``inf``
            when this sample is non-zero, which is the honest answer.

        Raises:
            ValueError: If the reference was computed on a different basis, since
                comparing a resolved-component index with a raw-signal one would
                be meaningless.
        """
        if reference.basis != self.basis:
            raise ValueError(
                f"cannot compare indices computed on different bases: "
                f"{self.basis!r} against {reference.basis!r}"
            )

        def ratio(current: float, base: float) -> float:
            if base == 0.0:
                return float("inf") if current > 0.0 else 1.0
            return current / base

        return {
            "carbonyl_index": ratio(self.carbonyl_index, reference.carbonyl_index),
            "acid_share": ratio(self.acid_share, reference.acid_share),
            "alkene_to_alkane": ratio(self.alkene_to_alkane, reference.alkene_to_alkane),
            "branching_index": ratio(self.branching_index, reference.branching_index),
        }

    def summary(self) -> dict[str, float | str | None]:
        """Flat, JSON-serialisable form for the recyclate passport."""
        return {
            "carbonyl_index": round(self.carbonyl_index, 5),
            "acid_share": round(self.acid_share, 5),
            "alkene_to_alkane": round(self.alkene_to_alkane, 5),
            "branching_index": round(self.branching_index, 5),
            "mean_carbon_number": (
                round(self.mean_carbon_number, 3)
                if self.mean_carbon_number is not None
                else None
            ),
            "chain_length_spread": (
                round(self.chain_length_spread, 3)
                if self.chain_length_spread is not None
                else None
            ),
            "basis": self.basis,
        }


def _ion_signal(cube: PyrogramDataCube, ions: tuple[float, ...]) -> float:
    """Total signal on a set of diagnostic ions."""
    available = [ion for ion in ions if np.min(np.abs(cube.mz_axis - ion)) <= 0.5]
    if not available:
        return 0.0
    return float(cube.eic_sum(available).sum())


def chain_length_statistics(
    model: PolyolefinMatrixModel,
) -> tuple[float, float]:
    """Area-weighted mean and spread of the comb's chain-length distribution.

    Read straight off the matrix model fitted in Milestone 2 — the amplitudes it
    produced per cluster *are* the chain-length distribution, so this needs no
    additional measurement.

    Note that the positions are the comb's relative series index, not absolute
    carbon numbers, unless the model's detection was anchored. Differences between
    samples measured with the same method are meaningful either way; absolute
    values need the anchor from the retention-index calibration.

    Args:
        model: Fitted polyolefin matrix model.

    Returns:
        ``(mean, standard deviation)`` over the series index.
    """
    amplitudes = np.asarray(model.amplitudes, dtype=np.float64)
    if amplitudes.size == 0 or amplitudes.sum() <= 0.0:
        return 0.0, 0.0

    # The model carries several segments per cluster; collapse them back onto
    # clusters before computing a chain-length distribution.
    n_clusters = model.detection.n_clusters
    if n_clusters > 0 and amplitudes.size % n_clusters == 0:
        per_cluster = amplitudes.reshape(n_clusters, -1).sum(axis=1)
    else:  # pragma: no cover - defensive
        per_cluster = amplitudes

    positions = np.arange(per_cluster.size, dtype=np.float64)
    weights = per_cluster / per_cluster.sum()
    mean = float(np.sum(positions * weights))
    spread = float(np.sqrt(np.sum(weights * (positions - mean) ** 2)))
    return mean, spread


def compute_degradation_indices(
    cube: PyrogramDataCube,
    *,
    matrix_model: PolyolefinMatrixModel | None = None,
    basis: str = "ion-signal",
) -> DegradationIndices:
    """Compute ageing indices from a preprocessed pyrogram.

    Args:
        cube: Baseline-corrected pyrogram. **Not** matrix-subtracted: the alkane
            signal is the denominator of three of the four ratios, and subtracting
            it first would divide by what is left of it.
        matrix_model: Fitted comb model, for the chain-length statistics. Optional;
            without it those two fields stay ``None``.
        basis: Label recording how the indices were derived, so that two sets are
            not compared across incompatible derivations.

    Returns:
        The indices, with caveats attached.

    Raises:
        ValueError: If the cube carries no signal at all.
    """
    if cube.total_signal <= 0.0:
        raise ValueError("cannot compute degradation indices on an empty pyrogram")

    aldehyde = _ion_signal(cube, OXIDATION_IONS["aldehyde"])
    ketone = _ion_signal(cube, OXIDATION_IONS["ketone"])
    acid = _ion_signal(cube, OXIDATION_IONS["acid"])
    oxidised = aldehyde + ketone + acid

    alkane = _ion_signal(cube, HYDROCARBON_IONS["alkane"])
    alkene = _ion_signal(cube, HYDROCARBON_IONS["alkene"])
    branched = _ion_signal(cube, HYDROCARBON_IONS["branched"])

    def safe_ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0.0 else 0.0

    notes: list[str] = [
        "indices are ratios and are interpretable only against a virgin reference "
        "of the same polymer measured under the same pyrolysis conditions",
    ]

    carbonyl_index = safe_ratio(oxidised, alkane)
    # The acid share divides one small number by another. On barely oxidised
    # material all three carbonyl channels read mostly background, and the share
    # collapses towards 1/3 regardless of the chemistry — measured at exactly
    # 0.333 on virgin PE. Below this carbonyl level the figure carries no
    # information, and it says so rather than being quietly reported.
    if carbonyl_index < _ACID_SHARE_MIN_CARBONYL:
        notes.append(
            f"carbonyl index {carbonyl_index:.4f} is at the background level; the "
            "acid share is not informative here and should not be interpreted"
        )
    if alkane <= 0.0:
        notes.append(
            "no alkane signal found; the sample does not look polyolefin-based and "
            "these indices do not apply to it"
        )

    mean_carbon: float | None = None
    spread: float | None = None
    if matrix_model is not None:
        mean_carbon, spread = chain_length_statistics(matrix_model)
        if not matrix_model.detection.contaminated.any():
            notes.append("chain-length statistics taken from the fitted comb model")

    return DegradationIndices(
        carbonyl_index=carbonyl_index,
        acid_share=safe_ratio(acid, oxidised),
        alkene_to_alkane=safe_ratio(alkene, alkane),
        branching_index=safe_ratio(branched, alkane),
        mean_carbon_number=mean_carbon,
        chain_length_spread=spread,
        basis=basis,
        notes=tuple(notes),
    )
