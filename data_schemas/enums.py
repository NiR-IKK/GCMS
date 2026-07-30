"""Controlled vocabularies for the Py-GC/MS recyclate domain.

Everything the platform reasons about — polymer classes, the role a compound
plays in a marker triad, recyclate stream provenance, raw-data formats — is
pinned to an explicit enumeration so that database rows, API payloads and
synthetic ground truth all speak the same language.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "PolymerClass",
    "MarkerRole",
    "RecyclateStream",
    "SourceFormat",
    "IonisationMode",
    "Polarity",
    "AcquisitionMode",
    "PyrolysisMode",
]


class PolymerClass(StrEnum):
    """Polymer (or non-polymeric) class a pyrolysis marker is attributed to.

    Values follow ISO 1043-1 / ISO 11469 abbreviations where these exist, so the
    identifiers can be re-used verbatim in a recyclate passport.
    """

    PE = "PE"
    """Polyethylene, grade not resolved."""
    PE_LD = "PE-LD"
    PE_HD = "PE-HD"
    PP = "PP"
    PS = "PS"
    ABS = "ABS"
    SAN = "SAN"
    PET = "PET"
    PBT = "PBT"
    PA6 = "PA6"
    PA66 = "PA66"
    PVC = "PVC"
    PC = "PC"
    PMMA = "PMMA"
    PLA = "PLA"
    EVA = "EVA"
    POM = "POM"
    PUR = "PUR"
    PPS = "PPS"
    NON_POLYMERIC = "non-polymeric"
    """Additives, processing aids, contaminants, column artefacts."""
    UNKNOWN = "unknown"

    @property
    def is_polyolefin(self) -> bool:
        """True for the aliphatic backbone classes that dominate PCR pyrograms."""
        return self in _POLYOLEFINS

    @property
    def is_styrenic(self) -> bool:
        return self in {PolymerClass.PS, PolymerClass.ABS, PolymerClass.SAN}

    @property
    def is_polar_condensation(self) -> bool:
        """Polyesters and polyamides — the trace fractions matrix subtraction targets."""
        return self in {
            PolymerClass.PET,
            PolymerClass.PBT,
            PolymerClass.PA6,
            PolymerClass.PA66,
            PolymerClass.PC,
            PolymerClass.PLA,
        }


_POLYOLEFINS = frozenset(
    {PolymerClass.PE, PolymerClass.PE_LD, PolymerClass.PE_HD, PolymerClass.PP, PolymerClass.EVA}
)


class MarkerRole(StrEnum):
    """Function of a compound inside a quantitative marker pattern.

    Polymer identification in this platform is never based on a single peak; it
    is based on ratios between roles (monomer : dimer : trimer, alkene : alkane,
    oxidised : saturated). The role therefore has to be first-class metadata.
    """

    MONOMER = "monomer"
    DIMER = "dimer"
    TRIMER = "trimer"
    TETRAMER = "tetramer"

    HOMOLOGUE_ALKANE = "homologue-n-alkane"
    HOMOLOGUE_ALKENE = "homologue-1-alkene"
    HOMOLOGUE_DIENE = "homologue-alkadiene"
    HOMOLOGUE_ISOALKENE = "homologue-iso-alkene"
    """Branched alkene — the PP fingerprint and a chain-scission indicator in PE."""

    OXIDATION_KETONE = "oxidation-ketone"
    OXIDATION_ALDEHYDE = "oxidation-aldehyde"
    OXIDATION_ACID = "oxidation-carboxylic-acid"

    ADDITIVE_ANTIOXIDANT = "additive-antioxidant"
    ADDITIVE_PLASTICISER = "additive-plasticiser"
    ADDITIVE_SLIP_AGENT = "additive-slip-agent"
    ADDITIVE_UV_STABILISER = "additive-uv-stabiliser"
    ADDITIVE_FLAME_RETARDANT = "additive-flame-retardant"

    CONTAMINANT = "contaminant"
    PYROLYSIS_FRAGMENT = "pyrolysis-fragment"
    COLUMN_BLEED = "column-bleed"

    @property
    def is_homologue(self) -> bool:
        """True for members of a homologous series (the polyolefin matrix)."""
        return self in _HOMOLOGUE_ROLES

    @property
    def is_oxidation_product(self) -> bool:
        """True for the oxygenated fragments that drive the degradation index."""
        return self in _OXIDATION_ROLES

    @property
    def is_additive(self) -> bool:
        return self.value.startswith("additive-")

    @property
    def is_oligomer(self) -> bool:
        """True for monomer/dimer/trimer/tetramer, i.e. members of a marker triad."""
        return self in _OLIGOMER_ROLES


_HOMOLOGUE_ROLES = frozenset(
    {
        MarkerRole.HOMOLOGUE_ALKANE,
        MarkerRole.HOMOLOGUE_ALKENE,
        MarkerRole.HOMOLOGUE_DIENE,
        MarkerRole.HOMOLOGUE_ISOALKENE,
    }
)

_OXIDATION_ROLES = frozenset(
    {
        MarkerRole.OXIDATION_KETONE,
        MarkerRole.OXIDATION_ALDEHYDE,
        MarkerRole.OXIDATION_ACID,
    }
)

_OLIGOMER_ROLES = frozenset(
    {MarkerRole.MONOMER, MarkerRole.DIMER, MarkerRole.TRIMER, MarkerRole.TETRAMER}
)


class RecyclateStream(StrEnum):
    """Provenance of the analysed material."""

    PCR_LDPE = "PCR-LDPE"
    PCR_HDPE = "PCR-HDPE"
    PCR_PP = "PCR-PP"
    PCR_PS = "PCR-PS"
    PCR_PET = "PCR-PET"
    PCR_MIXED_POLYOLEFIN = "PCR-mixed-polyolefin"
    """Sorting fraction 'DKR 324'-like: PE/PP dominated with foreign-polymer traces."""
    PCR_WEEE = "PCR-WEEE"
    PCR_ELV = "PCR-ELV"
    PIR = "PIR"
    """Post-industrial recyclate."""
    VIRGIN = "virgin"
    LAB_BLEND = "lab-blend"
    """Gravimetrically prepared reference blend — the validation ground truth."""
    UNKNOWN = "unknown"


class SourceFormat(StrEnum):
    """Raw-data container the pyrogram was ingested from."""

    MZML = "mzML"
    MZXML = "mzXML"
    MZDATA = "mzData"
    ANDI_CDF = "ANDI-CDF"
    """AIA/ANDI-MS netCDF (ASTM E1947) — the GC/MS vendor-neutral exchange format."""
    SYNTHETIC = "synthetic"
    """Generated by the SyntheticPyrogramGenerator, not measured."""


class IonisationMode(StrEnum):
    EI = "EI"
    CI = "CI"
    FI = "FI"
    UNKNOWN = "unknown"


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class AcquisitionMode(StrEnum):
    FULL_SCAN = "full-scan"
    SIM = "SIM"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class PyrolysisMode(StrEnum):
    """Sampling technique on the micro-furnace / filament pyrolyser."""

    SINGLE_SHOT = "single-shot"
    DOUBLE_SHOT = "double-shot"
    """Thermal desorption of additives followed by pyrolysis of the backbone."""
    EVOLVED_GAS = "evolved-gas-analysis"
    THERMAL_DESORPTION = "thermal-desorption"
    REACTIVE_THM = "reactive-pyrolysis-THM"
    """Thermally assisted hydrolysis and methylation (TMAH) for polar polymers."""
    UNKNOWN = "unknown"
