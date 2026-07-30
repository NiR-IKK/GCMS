"""Electron-impact mass spectra and retention anchors for pyrolysis markers.

This module is the *physical truth* of the simulator: it says what a compound
looks like in a 70 eV EI source and roughly where it elutes on a non-polar
5 %-phenyl column under the reference oven program. It is deliberately kept
inside ``tests/`` and separate from any identification library the platform
itself will ship, so that a future marker library cannot be validated against the
very table it was derived from.

Two kinds of entry live here:

* **Homologous series** (``alkane_spectrum`` and friends) are generated from
  fragmentation rules rather than tabulated. There are hundreds of them in a real
  polyolefin pyrogram, and — importantly — consecutive members have almost
  identical spectra. That near-collinearity is the central numerical difficulty
  of resolving a polyolefin matrix, so the simulator has to reproduce it rather
  than hand the algorithm artificially distinct spectra.
* **Discrete markers** (:data:`REFERENCE_COMPOUNDS`) are tabulated from the
  literature-typical fragment patterns of the specific compounds a recyclate lab
  looks for: the styrene triad, caprolactam for PA6, benzoic acid for PET,
  benzene/naphthalene for PVC, plus the usual additive package.

Intensities are relative (base peak = 100). Retention times are given in seconds
on the reference method described by :data:`REFERENCE_METHOD_WINDOW_S`; the
generator rescales them if it is configured with a different run length.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from data_schemas.enums import MarkerRole, PolymerClass

__all__ = [
    "REFERENCE_METHOD_WINDOW_S",
    "ReferenceCompound",
    "REFERENCE_COMPOUNDS",
    "SILOXANE_BLEED_SPECTRUM",
    "alkane_spectrum",
    "alkene_spectrum",
    "alkadiene_spectrum",
    "isoalkene_spectrum",
    "alkanone_spectrum",
    "alkanal_spectrum",
    "alkanoic_acid_spectrum",
    "homologue_spectrum",
    "homologue_molecular_weight",
]

REFERENCE_METHOD_WINDOW_S: tuple[float, float] = (60.0, 1800.0)
"""Retention window all tabulated retention times refer to.

Corresponds to 40 °C (2 min) ramped at 10 °C/min to 320 °C with a terminal hold,
on a 30 m x 0.25 mm x 0.25 um 5 % diphenyl / 95 % dimethylpolysiloxane column —
the de-facto standard method for polymer pyrolysates.
"""


@dataclass(frozen=True, slots=True)
class ReferenceCompound:
    """A tabulated pyrolysis marker.

    Attributes:
        name: Compound name as used in reports and marker-library keys.
        molecular_weight: Nominal molecular mass in Da.
        spectrum: m/z to relative intensity (base peak = 100).
        retention_time_s: Apex on the reference method, in seconds.
        polymer_class: Material this compound is diagnostic for.
        role: Function inside its marker pattern.
        quantifier_mz: Ion to integrate for quantification — chosen for
            selectivity against the polyolefin background, which is why it is not
            always the base peak.
        peak_width_scale: Peak width relative to the method's base width. Polar or
            high-boiling compounds are broader.
        tailing_scale: Tailing relative to the method's base tailing. Acids,
            amides and lactams tail badly on a non-polar phase.
    """

    name: str
    molecular_weight: float
    spectrum: Mapping[int, float]
    retention_time_s: float
    polymer_class: PolymerClass
    role: MarkerRole
    quantifier_mz: int
    peak_width_scale: float = 1.0
    tailing_scale: float = 1.0


# ---------------------------------------------------------------------------
# Homologous-series fragmentation rules
# ---------------------------------------------------------------------------

# Relative abundance of the C_kH_{2k+1}+ alkyl ion series (m/z = 14k + 1).
# Peaks at m/z 43 (k=3) / 57 (k=4) and decays geometrically, as observed for
# every n-alkane above about C8.
_ALKYL_WEIGHTS: dict[int, float] = {1: 8.0, 2: 28.0, 3: 100.0, 4: 92.0, 5: 55.0, 6: 35.0}
_ALKYL_DECAY = 0.62

# Relative abundance of the C_kH_{2k-1}+ alkenyl series (m/z = 14k - 1).
_ALKENYL_WEIGHTS: dict[int, float] = {3: 100.0, 4: 78.0, 5: 48.0, 6: 30.0}
_ALKENYL_DECAY = 0.60


def _alkyl_series(max_carbon: int) -> dict[int, float]:
    """Alkyl fragment ions ``C_kH_{2k+1}+`` up to ``max_carbon``."""
    series: dict[int, float] = {}
    for k in range(1, max_carbon + 1):
        weight = _ALKYL_WEIGHTS.get(k)
        if weight is None:
            weight = _ALKYL_WEIGHTS[6] * _ALKYL_DECAY ** (k - 6)
        if weight < 0.05:
            break
        series[14 * k + 1] = weight
    return series


def _alkenyl_series(max_carbon: int, *, scale: float = 1.0) -> dict[int, float]:
    """Alkenyl fragment ions ``C_kH_{2k-1}+`` up to ``max_carbon``."""
    series: dict[int, float] = {}
    for k in range(2, max_carbon + 1):
        weight = _ALKENYL_WEIGHTS.get(k)
        if weight is None:
            weight = (
                _ALKENYL_WEIGHTS[6] * _ALKENYL_DECAY ** (k - 6)
                if k > 6
                else _ALKENYL_WEIGHTS[3] * 0.45
            )
        weight *= scale
        if weight < 0.05:
            break
        series[14 * k - 1] = weight
    return series


def _merge(*parts: Mapping[int, float]) -> dict[int, float]:
    """Sum several ion dictionaries into one."""
    merged: dict[int, float] = {}
    for part in parts:
        for mz, intensity in part.items():
            merged[mz] = merged.get(mz, 0.0) + intensity
    return merged


def _finalise(spectrum: Mapping[int, float], molecular_weight: int) -> dict[int, float]:
    """Drop ions above the molecular ion and renormalise to base peak = 100.

    A fragment cannot be heavier than the molecule it came from; the generated
    series are truncated accordingly, which is what makes short-chain and
    long-chain homologues differ at all.
    """
    trimmed = {
        mz: intensity
        for mz, intensity in spectrum.items()
        if 12 <= mz <= molecular_weight and intensity > 0.0
    }
    if not trimmed:
        return {molecular_weight: 100.0}
    peak = max(trimmed.values())
    return {mz: 100.0 * intensity / peak for mz, intensity in sorted(trimmed.items())}


def homologue_molecular_weight(kind: str, carbon_number: int) -> int:
    """Nominal molecular mass of a homologous-series member.

    Args:
        kind: One of ``"alkane"``, ``"alkene"``, ``"diene"``, ``"isoalkene"``,
            ``"alkanone"``, ``"alkanal"``, ``"acid"``.
        carbon_number: Number of carbon atoms.

    Returns:
        Nominal molecular mass in Da.

    Raises:
        ValueError: If ``kind`` is unknown or ``carbon_number`` is too small.
    """
    if carbon_number < 1:
        raise ValueError(f"carbon_number must be >= 1, got {carbon_number}")
    match kind:
        case "alkane":
            return 14 * carbon_number + 2
        case "alkene" | "isoalkene":
            return 14 * carbon_number
        case "diene":
            return 14 * carbon_number - 2
        case "alkanone" | "alkanal":
            return 14 * carbon_number
        case "acid":
            return 14 * carbon_number + 2
        case _:
            raise ValueError(f"unknown homologue kind {kind!r}")


def alkane_spectrum(carbon_number: int) -> dict[int, float]:
    """EI spectrum of an n-alkane ``C_nH_{2n+2}``.

    Dominated by the alkyl series with 43/57 as base peaks, a weaker alkenyl
    series from hydrogen loss, and a small molecular ion that fades with chain
    length. This is the single most abundant pattern in any polyolefin pyrogram.

    Args:
        carbon_number: Chain length ``n``, at least 2.

    Returns:
        m/z to relative intensity, base peak normalised to 100.

    Raises:
        ValueError: If ``carbon_number`` is below 2.
    """
    if carbon_number < 2:
        raise ValueError(f"n-alkane needs at least 2 carbons, got {carbon_number}")
    molecular_weight = homologue_molecular_weight("alkane", carbon_number)
    molecular_ion = max(0.6, 14.0 * 0.90**carbon_number)
    parts = [
        _alkyl_series(carbon_number),
        _alkenyl_series(carbon_number, scale=0.33),
        {molecular_weight: molecular_ion},
        {molecular_weight - 15: molecular_ion * 0.25},
        {molecular_weight - 29: molecular_ion * 0.18},
    ]
    return _finalise(_merge(*parts), molecular_weight)


def alkene_spectrum(carbon_number: int) -> dict[int, float]:
    """EI spectrum of a 1-alkene ``C_nH_{2n}``.

    Allylic cleavage makes the alkenyl series (41/55/69) dominant and adds the
    even-mass ``C_kH_{2k}+`` ions that distinguish an alkene from the alkane of
    the same carbon number — the basis of the alkene/alkane degradation ratio.

    Args:
        carbon_number: Chain length ``n``, at least 3.

    Returns:
        m/z to relative intensity, base peak normalised to 100.

    Raises:
        ValueError: If ``carbon_number`` is below 3.
    """
    if carbon_number < 3:
        raise ValueError(f"1-alkene needs at least 3 carbons, got {carbon_number}")
    molecular_weight = homologue_molecular_weight("alkene", carbon_number)
    molecular_ion = max(1.5, 22.0 * 0.93**carbon_number)
    even_mass = {
        14 * k: 34.0 * _ALKENYL_DECAY ** max(0, k - 4)
        for k in range(3, min(carbon_number, 9) + 1)
    }
    # The alkyl series is scaled down: an alkene fragments allylically first, so
    # 41/55 outrun 43/57 — the inverse of the alkane pattern.
    parts = _merge(
        _alkenyl_series(carbon_number),
        {mz: value * 0.30 for mz, value in _alkyl_series(carbon_number).items()},
        even_mass,
        {molecular_weight: molecular_ion},
        {molecular_weight - 14: molecular_ion * 0.4},
    )
    return _finalise(parts, molecular_weight)


def alkadiene_spectrum(carbon_number: int) -> dict[int, float]:
    """EI spectrum of an alkadiene ``C_nH_{2n-2}``.

    Dienes appear in polyolefin pyrolysates through intramolecular hydrogen
    transfer and are the third member of the characteristic PE triplet. Their
    diagnostic ions are the odd ``C_kH_{2k-3}+`` series (67, 81, 95).

    Args:
        carbon_number: Chain length ``n``, at least 4.

    Returns:
        m/z to relative intensity, base peak normalised to 100.

    Raises:
        ValueError: If ``carbon_number`` is below 4.
    """
    if carbon_number < 4:
        raise ValueError(f"alkadiene needs at least 4 carbons, got {carbon_number}")
    molecular_weight = homologue_molecular_weight("diene", carbon_number)
    dienyl = {
        14 * k - 3: 100.0 * 0.58 ** max(0, k - 5)
        for k in range(4, min(carbon_number, 10) + 1)
    }
    parts = _merge(
        dienyl,
        {mz: value * 0.55 for mz, value in _alkenyl_series(carbon_number).items()},
        {14 * k - 2: 30.0 * 0.6 ** max(0, k - 4) for k in range(4, min(carbon_number, 8) + 1)},
        {molecular_weight: max(2.0, 26.0 * 0.94**carbon_number)},
    )
    return _finalise(parts, molecular_weight)


def isoalkene_spectrum(carbon_number: int) -> dict[int, float]:
    """EI spectrum of a branched (iso) alkene ``C_nH_{2n}``.

    Branch points direct allylic cleavage to give strong even-mass ions (56, 70,
    84). The 70/71 intensity ratio therefore separates branched from linear
    material, which is what the branching index in the recyclate passport uses:
    PP pyrolysates and LDPE short-chain branches are rich in these, HDPE is not.

    Args:
        carbon_number: Chain length ``n``, at least 6.

    Returns:
        m/z to relative intensity, base peak normalised to 100.

    Raises:
        ValueError: If ``carbon_number`` is below 6.
    """
    if carbon_number < 6:
        raise ValueError(f"iso-alkene needs at least 6 carbons, got {carbon_number}")
    molecular_weight = homologue_molecular_weight("isoalkene", carbon_number)
    branched_even = {
        14 * k: 100.0 * 0.55 ** abs(k - 5)
        for k in range(3, min(carbon_number, 10) + 1)
    }
    parts = _merge(
        branched_even,
        {mz: value * 0.55 for mz, value in _alkenyl_series(carbon_number).items()},
        {mz: value * 0.35 for mz, value in _alkyl_series(carbon_number).items()},
        {molecular_weight: max(2.0, 18.0 * 0.93**carbon_number)},
        {molecular_weight - 15: max(1.0, 10.0 * 0.93**carbon_number)},
    )
    return _finalise(parts, molecular_weight)


def alkanone_spectrum(carbon_number: int) -> dict[int, float]:
    """EI spectrum of a 2-alkanone ``C_nH_{2n}O``.

    Methyl ketones are the signature thermo-oxidative degradation product of
    polyolefins. The McLafferty rearrangement ion at m/z 58 plus the acetyl ion
    at m/z 43 make them detectable against a huge hydrocarbon background, and
    their abundance drives the carbonyl term of the degradation index.

    Args:
        carbon_number: Chain length ``n``, at least 4.

    Returns:
        m/z to relative intensity, base peak normalised to 100.

    Raises:
        ValueError: If ``carbon_number`` is below 4.
    """
    if carbon_number < 4:
        raise ValueError(f"2-alkanone needs at least 4 carbons, got {carbon_number}")
    molecular_weight = homologue_molecular_weight("alkanone", carbon_number)
    # The acetyl ion at m/z 43 competes closely with the McLafferty ion at 58 but
    # stays below it; the tabulated weight allows for the alkyl series adding to 43.
    parts = _merge(
        {58: 100.0, 43: 72.0, 59: 26.0, 71: 12.0, 42: 9.0, 41: 14.0},
        {mz: value * 0.16 for mz, value in _alkyl_series(carbon_number).items()},
        {molecular_weight: max(1.5, 12.0 * 0.94**carbon_number)},
        {molecular_weight - 15: 4.0},
        {molecular_weight - 43: 8.0},
    )
    return _finalise(parts, molecular_weight)


def alkanal_spectrum(carbon_number: int) -> dict[int, float]:
    """EI spectrum of an n-alkanal ``C_nH_{2n}O``.

    Aldehydes are the early oxidation intermediates that precede ketones and
    acids; m/z 44 is their McLafferty ion.

    Args:
        carbon_number: Chain length ``n``, at least 4.

    Returns:
        m/z to relative intensity, base peak normalised to 100.

    Raises:
        ValueError: If ``carbon_number`` is below 4.
    """
    if carbon_number < 4:
        raise ValueError(f"n-alkanal needs at least 4 carbons, got {carbon_number}")
    molecular_weight = homologue_molecular_weight("alkanal", carbon_number)
    parts = _merge(
        {44: 100.0, 43: 66.0, 41: 54.0, 29: 48.0, 57: 34.0, 55: 30.0, 45: 12.0},
        {mz: value * 0.22 for mz, value in _alkyl_series(carbon_number).items()},
        {molecular_weight: max(1.0, 9.0 * 0.94**carbon_number)},
        {molecular_weight - 1: 7.0},
        {molecular_weight - 44: 10.0},
    )
    return _finalise(parts, molecular_weight)


def alkanoic_acid_spectrum(carbon_number: int) -> dict[int, float]:
    """EI spectrum of an n-alkanoic acid ``C_nH_{2n}O2``.

    Carboxylic acids are the terminal oxidation products and the strongest
    indicator of advanced ageing. Their m/z 60 / 73 pair is highly selective
    against hydrocarbons.

    Args:
        carbon_number: Chain length ``n``, at least 4.

    Returns:
        m/z to relative intensity, base peak normalised to 100.

    Raises:
        ValueError: If ``carbon_number`` is below 4.
    """
    if carbon_number < 4:
        raise ValueError(f"n-alkanoic acid needs at least 4 carbons, got {carbon_number}")
    molecular_weight = homologue_molecular_weight("acid", carbon_number)
    parts = _merge(
        {60: 100.0, 73: 82.0, 43: 52.0, 41: 46.0, 57: 28.0, 55: 26.0, 45: 14.0, 29: 20.0},
        {mz: value * 0.18 for mz, value in _alkyl_series(carbon_number).items()},
        {molecular_weight: max(2.0, 20.0 * 0.96**carbon_number)},
        {molecular_weight - 17: 5.0},
        {molecular_weight - 18: 4.0},
    )
    return _finalise(parts, molecular_weight)


_HOMOLOGUE_BUILDERS = MappingProxyType(
    {
        "alkane": alkane_spectrum,
        "alkene": alkene_spectrum,
        "diene": alkadiene_spectrum,
        "isoalkene": isoalkene_spectrum,
        "alkanone": alkanone_spectrum,
        "alkanal": alkanal_spectrum,
        "acid": alkanoic_acid_spectrum,
    }
)

_HOMOLOGUE_ROLES = MappingProxyType(
    {
        "alkane": MarkerRole.HOMOLOGUE_ALKANE,
        "alkene": MarkerRole.HOMOLOGUE_ALKENE,
        "diene": MarkerRole.HOMOLOGUE_DIENE,
        "isoalkene": MarkerRole.HOMOLOGUE_ISOALKENE,
        "alkanone": MarkerRole.OXIDATION_KETONE,
        "alkanal": MarkerRole.OXIDATION_ALDEHYDE,
        "acid": MarkerRole.OXIDATION_ACID,
    }
)

_HOMOLOGUE_QUANTIFIER_MZ = MappingProxyType(
    {
        "alkane": 57,
        "alkene": 55,
        "diene": 67,
        "isoalkene": 70,
        "alkanone": 58,
        "alkanal": 44,
        "acid": 60,
    }
)


def homologue_spectrum(kind: str, carbon_number: int) -> dict[int, float]:
    """Dispatch to the spectrum generator for a homologous series.

    Args:
        kind: One of ``"alkane"``, ``"alkene"``, ``"diene"``, ``"isoalkene"``,
            ``"alkanone"``, ``"alkanal"``, ``"acid"``.
        carbon_number: Chain length.

    Returns:
        m/z to relative intensity, base peak normalised to 100.

    Raises:
        ValueError: If ``kind`` is unknown or the chain is too short for it.
    """
    builder = _HOMOLOGUE_BUILDERS.get(kind)
    if builder is None:
        raise ValueError(
            f"unknown homologue kind {kind!r}; expected one of "
            f"{', '.join(sorted(_HOMOLOGUE_BUILDERS))}"
        )
    return builder(carbon_number)


def homologue_role(kind: str) -> MarkerRole:
    """Marker role associated with a homologous series kind."""
    role = _HOMOLOGUE_ROLES.get(kind)
    if role is None:
        raise ValueError(f"unknown homologue kind {kind!r}")
    return role


def homologue_quantifier_mz(kind: str) -> int:
    """Selective quantifier ion for a homologous series kind."""
    quantifier = _HOMOLOGUE_QUANTIFIER_MZ.get(kind)
    if quantifier is None:
        raise ValueError(f"unknown homologue kind {kind!r}")
    return quantifier


# ---------------------------------------------------------------------------
# Column bleed
# ---------------------------------------------------------------------------

SILOXANE_BLEED_SPECTRUM: Mapping[int, float] = MappingProxyType(
    {73: 100.0, 147: 72.0, 207: 96.0, 221: 34.0, 281: 28.0, 355: 11.0}
)
"""Cyclic siloxane ions of a degrading dimethylpolysiloxane phase.

Reproducing bleed on exactly these channels — rather than as a flat offset — is
what makes the baseline problem realistic: a trace PET marker at m/z 149 sits
next to the 147 bleed ion, and a naive TIC-based baseline correction gets it wrong.
"""


# ---------------------------------------------------------------------------
# Discrete pyrolysis markers
# ---------------------------------------------------------------------------


def _compound(
    name: str,
    molecular_weight: float,
    spectrum: dict[int, float],
    retention_time_s: float,
    polymer_class: PolymerClass,
    role: MarkerRole,
    quantifier_mz: int,
    peak_width_scale: float = 1.0,
    tailing_scale: float = 1.0,
) -> tuple[str, ReferenceCompound]:
    """Build a ``(name, ReferenceCompound)`` pair for the registry."""
    return name, ReferenceCompound(
        name=name,
        molecular_weight=molecular_weight,
        spectrum=MappingProxyType(dict(sorted(spectrum.items()))),
        retention_time_s=retention_time_s,
        polymer_class=polymer_class,
        role=role,
        quantifier_mz=quantifier_mz,
        peak_width_scale=peak_width_scale,
        tailing_scale=tailing_scale,
    )


REFERENCE_COMPOUNDS: Mapping[str, ReferenceCompound] = MappingProxyType(
    dict(
        [
            # -------------------------------------------------- PVC / aromatics
            _compound(
                "hydrogen chloride",
                36.0,
                {36: 100.0, 38: 32.0, 35: 16.0},
                92.0,
                PolymerClass.PVC,
                MarkerRole.PYROLYSIS_FRAGMENT,
                36,
                peak_width_scale=1.6,
                tailing_scale=3.0,
            ),
            _compound(
                "benzene",
                78.0,
                {78: 100.0, 77: 24.0, 52: 19.0, 51: 24.0, 50: 15.0, 39: 8.0},
                248.0,
                PolymerClass.PVC,
                MarkerRole.PYROLYSIS_FRAGMENT,
                78,
            ),
            _compound(
                "toluene",
                92.0,
                {91: 100.0, 92: 68.0, 65: 12.0, 51: 6.0, 39: 9.0},
                332.0,
                PolymerClass.PVC,
                MarkerRole.PYROLYSIS_FRAGMENT,
                91,
            ),
            _compound(
                "indene",
                116.0,
                {116: 100.0, 115: 92.0, 89: 11.0, 63: 9.0, 39: 7.0},
                482.0,
                PolymerClass.PVC,
                MarkerRole.PYROLYSIS_FRAGMENT,
                116,
            ),
            _compound(
                "naphthalene",
                128.0,
                {128: 100.0, 127: 17.0, 129: 9.0, 102: 8.0, 64: 6.0, 51: 7.0},
                612.0,
                PolymerClass.PVC,
                MarkerRole.PYROLYSIS_FRAGMENT,
                128,
            ),
            _compound(
                "chlorobenzene",
                112.0,
                {112: 100.0, 77: 92.0, 114: 32.0, 51: 24.0, 50: 18.0},
                388.0,
                PolymerClass.PVC,
                MarkerRole.PYROLYSIS_FRAGMENT,
                112,
            ),
            # -------------------------------------------------------- styrenics
            _compound(
                "styrene",
                104.0,
                {104: 100.0, 103: 62.0, 78: 46.0, 51: 31.0, 77: 17.0, 102: 12.0, 50: 12.0},
                414.0,
                PolymerClass.PS,
                MarkerRole.MONOMER,
                104,
            ),
            _compound(
                "alpha-methylstyrene",
                118.0,
                {118: 100.0, 117: 93.0, 115: 30.0, 103: 38.0, 91: 33.0, 78: 18.0, 51: 11.0},
                468.0,
                PolymerClass.PS,
                MarkerRole.PYROLYSIS_FRAGMENT,
                118,
            ),
            _compound(
                "2,4-diphenyl-1-butene",
                208.0,
                {91: 100.0, 104: 58.0, 117: 34.0, 208: 24.0, 130: 15.0, 193: 10.0, 78: 11.0},
                1146.0,
                PolymerClass.PS,
                MarkerRole.DIMER,
                208,
                peak_width_scale=1.15,
            ),
            _compound(
                "2,4,6-triphenyl-1-hexene",
                312.0,
                {91: 100.0, 117: 44.0, 104: 39.0, 194: 20.0, 207: 14.0, 130: 12.0, 312: 5.0},
                1522.0,
                PolymerClass.PS,
                MarkerRole.TRIMER,
                194,
                peak_width_scale=1.35,
                tailing_scale=1.4,
            ),
            _compound(
                "benzaldehyde",
                106.0,
                {106: 100.0, 105: 96.0, 77: 88.0, 51: 34.0, 50: 18.0},
                452.0,
                PolymerClass.PS,
                MarkerRole.OXIDATION_ALDEHYDE,
                105,
                tailing_scale=1.5,
            ),
            _compound(
                "acetophenone",
                120.0,
                {105: 100.0, 77: 62.0, 120: 30.0, 51: 22.0, 43: 12.0},
                560.0,
                PolymerClass.PS,
                MarkerRole.OXIDATION_KETONE,
                105,
                tailing_scale=1.5,
            ),
            _compound(
                "acrylonitrile",
                53.0,
                {53: 100.0, 52: 74.0, 26: 58.0, 51: 28.0, 27: 20.0},
                152.0,
                PolymerClass.SAN,
                MarkerRole.MONOMER,
                53,
            ),
            _compound(
                "4-vinylcyclohexene",
                108.0,
                {54: 100.0, 79: 62.0, 108: 40.0, 67: 30.0, 39: 22.0},
                358.0,
                PolymerClass.ABS,
                MarkerRole.DIMER,
                54,
            ),
            # ------------------------------------------------------ polypropene
            _compound(
                "2,4-dimethyl-1-heptene",
                126.0,
                {70: 100.0, 41: 42.0, 43: 26.0, 55: 34.0, 56: 21.0, 69: 18.0, 83: 10.0, 126: 12.0},
                454.0,
                PolymerClass.PP,
                MarkerRole.TRIMER,
                126,
            ),
            # ------------------------------------------------------------- PET
            _compound(
                "benzoic acid",
                122.0,
                {105: 100.0, 122: 84.0, 77: 72.0, 51: 28.0, 50: 14.0},
                642.0,
                PolymerClass.PET,
                MarkerRole.PYROLYSIS_FRAGMENT,
                105,
                peak_width_scale=1.2,
                tailing_scale=2.6,
            ),
            _compound(
                "vinyl benzoate",
                148.0,
                {105: 100.0, 77: 58.0, 148: 21.0, 51: 24.0, 27: 14.0},
                726.0,
                PolymerClass.PET,
                MarkerRole.MONOMER,
                148,
                tailing_scale=1.4,
            ),
            _compound(
                "divinyl terephthalate",
                218.0,
                {149: 100.0, 175: 29.0, 218: 15.0, 121: 24.0, 76: 15.0, 65: 18.0},
                1012.0,
                PolymerClass.PET,
                MarkerRole.DIMER,
                149,
                peak_width_scale=1.2,
                tailing_scale=1.6,
            ),
            _compound(
                "biphenyl",
                154.0,
                {154: 100.0, 153: 44.0, 152: 29.0, 76: 14.0, 51: 8.0},
                748.0,
                PolymerClass.PET,
                MarkerRole.PYROLYSIS_FRAGMENT,
                154,
            ),
            # ------------------------------------------------------------- PA6
            _compound(
                "epsilon-caprolactam",
                113.0,
                {113: 100.0, 85: 74.0, 55: 58.0, 56: 54.0, 30: 44.0, 84: 29.0, 42: 24.0, 41: 19.0},
                694.0,
                PolymerClass.PA6,
                MarkerRole.MONOMER,
                113,
                peak_width_scale=1.25,
                tailing_scale=2.8,
            ),
            _compound(
                "hexanenitrile",
                97.0,
                {97: 100.0, 54: 68.0, 41: 60.0, 82: 30.0, 43: 26.0},
                462.0,
                PolymerClass.PA6,
                MarkerRole.PYROLYSIS_FRAGMENT,
                97,
                tailing_scale=1.4,
            ),
            _compound(
                "cyclopentanone",
                84.0,
                {84: 100.0, 55: 92.0, 56: 42.0, 41: 34.0, 39: 24.0},
                338.0,
                PolymerClass.PA66,
                MarkerRole.PYROLYSIS_FRAGMENT,
                84,
            ),
            # -------------------------------------------------------------- PC
            _compound(
                "bisphenol A",
                228.0,
                {213: 100.0, 228: 34.0, 119: 24.0, 135: 11.0, 91: 12.0},
                1182.0,
                PolymerClass.PC,
                MarkerRole.MONOMER,
                213,
                peak_width_scale=1.3,
                tailing_scale=2.4,
            ),
            _compound(
                "phenol",
                94.0,
                {94: 100.0, 66: 24.0, 65: 18.0, 39: 14.0},
                404.0,
                PolymerClass.PC,
                MarkerRole.PYROLYSIS_FRAGMENT,
                94,
                tailing_scale=2.2,
            ),
            _compound(
                "4-isopropenylphenol",
                134.0,
                {134: 100.0, 119: 68.0, 91: 30.0, 77: 18.0},
                646.0,
                PolymerClass.PC,
                MarkerRole.PYROLYSIS_FRAGMENT,
                134,
                tailing_scale=2.0,
            ),
            # ------------------------------------------------------------ PMMA
            _compound(
                "methyl methacrylate",
                100.0,
                {69: 100.0, 100: 44.0, 41: 34.0, 39: 30.0, 59: 14.0},
                236.0,
                PolymerClass.PMMA,
                MarkerRole.MONOMER,
                100,
            ),
            _compound(
                "methyl methacrylate dimer",
                200.0,
                {69: 100.0, 41: 44.0, 141: 22.0, 100: 20.0, 200: 4.0},
                708.0,
                PolymerClass.PMMA,
                MarkerRole.DIMER,
                141,
            ),
            # -------------------------------------------------------- additives
            _compound(
                "2,6-di-tert-butyl-4-methylphenol (BHT)",
                220.0,
                {205: 100.0, 220: 24.0, 177: 12.0, 145: 9.0, 57: 14.0, 81: 7.0},
                702.0,
                PolymerClass.NON_POLYMERIC,
                MarkerRole.ADDITIVE_ANTIOXIDANT,
                205,
            ),
            _compound(
                "2,4-di-tert-butylphenol",
                206.0,
                {191: 100.0, 206: 33.0, 57: 24.0, 147: 14.0, 163: 10.0},
                802.0,
                PolymerClass.NON_POLYMERIC,
                MarkerRole.ADDITIVE_ANTIOXIDANT,
                191,
                tailing_scale=1.6,
            ),
            _compound(
                "bis(2-ethylhexyl) phthalate (DEHP)",
                390.0,
                {149: 100.0, 167: 29.0, 279: 24.0, 57: 18.0, 71: 14.0, 113: 11.0, 43: 19.0},
                1252.0,
                PolymerClass.NON_POLYMERIC,
                MarkerRole.ADDITIVE_PLASTICISER,
                149,
                peak_width_scale=1.35,
                tailing_scale=1.8,
            ),
            _compound(
                "dibutyl phthalate (DBP)",
                278.0,
                {149: 100.0, 205: 12.0, 223: 9.0, 121: 8.0, 41: 14.0, 278: 3.0},
                1024.0,
                PolymerClass.NON_POLYMERIC,
                MarkerRole.ADDITIVE_PLASTICISER,
                149,
                peak_width_scale=1.2,
                tailing_scale=1.7,
            ),
            _compound(
                "octadecanoic acid (stearic acid)",
                284.0,
                {60: 100.0, 73: 84.0, 43: 56.0, 129: 34.0, 185: 19.0, 284: 17.0, 41: 44.0},
                1124.0,
                PolymerClass.NON_POLYMERIC,
                MarkerRole.ADDITIVE_SLIP_AGENT,
                60,
                peak_width_scale=1.25,
                tailing_scale=3.2,
            ),
            _compound(
                "erucamide",
                337.0,
                {59: 100.0, 72: 58.0, 55: 48.0, 41: 44.0, 126: 10.0, 69: 30.0},
                1292.0,
                PolymerClass.NON_POLYMERIC,
                MarkerRole.ADDITIVE_SLIP_AGENT,
                59,
                peak_width_scale=1.4,
                tailing_scale=3.4,
            ),
            _compound(
                "2-(2H-benzotriazol-2-yl)-p-cresol fragment",
                225.0,
                {225: 100.0, 120: 46.0, 92: 28.0, 65: 18.0, 197: 22.0},
                1146.0,
                PolymerClass.NON_POLYMERIC,
                MarkerRole.ADDITIVE_UV_STABILISER,
                225,
                peak_width_scale=1.3,
                tailing_scale=2.0,
            ),
            _compound(
                "tetrabromobisphenol A fragment",
                250.0,
                {250: 100.0, 252: 96.0, 171: 30.0, 143: 22.0, 63: 18.0},
                1338.0,
                PolymerClass.NON_POLYMERIC,
                MarkerRole.ADDITIVE_FLAME_RETARDANT,
                250,
                peak_width_scale=1.3,
                tailing_scale=2.2,
            ),
        ]
    )
)
"""Registry of tabulated markers, keyed by compound name."""
