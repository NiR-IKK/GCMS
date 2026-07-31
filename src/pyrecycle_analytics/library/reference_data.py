"""Curated reference data for the marker library.

**This table is deliberately independent of ``tests/reference_spectra.py``.**

That separation is the single most important methodological property of Milestone
3. The synthetic benchmark generates its pyrograms from one spectral table; if the
identification library were derived from the same table, every identification
score would be measuring the library against itself and would prove nothing. The
values here come from the published EI fragmentation patterns of these compounds
and from tabulated Kováts indices on non-polar 5 %-phenyl phases — the same
sources a laboratory would use to build a real library.

The practical consequence is that identification scores on the benchmark come out
*lower* than they would with a self-consistent table. That is the intended
outcome. A test asserts that the two tables are not identical, so the separation
cannot be lost by accident later.

Retention indices are given for a 5 % diphenyl / 95 % dimethylpolysiloxane phase.
They are what makes the library transferable: retention time depends on column
length, flow and oven program, the Kováts index largely does not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from data_schemas.enums import MarkerRole, PolymerClass

__all__ = [
    "LibraryCompound",
    "LibraryPattern",
    "LibraryPatternMember",
    "REFERENCE_COMPOUNDS",
    "MARKER_PATTERNS",
    "RESPONSE_FACTORS",
    "STATIONARY_PHASE",
]

STATIONARY_PHASE = "5%-phenyl"
"""Phase all tabulated retention indices refer to."""


@dataclass(frozen=True, slots=True)
class LibraryCompound:
    """A reference compound as the library stores it."""

    name: str
    molecular_weight: float
    quantifier_mz: int
    retention_index: float
    spectrum: Mapping[int, float]
    cas_number: str | None = None
    formula: str | None = None
    ri_tolerance: float = 40.0
    source: str = "literature EI pattern; Kovats index on 5%-phenyl"


@dataclass(frozen=True, slots=True)
class LibraryPatternMember:
    """One member of a marker pattern."""

    compound: str
    role: MarkerRole
    relative_abundance: float
    is_reference: bool = False
    tolerance_factor: float = 2.5


@dataclass(frozen=True, slots=True)
class LibraryPattern:
    """A quantitative marker pattern identifying one polymer."""

    name: str
    polymer: PolymerClass
    description: str
    members: tuple[LibraryPatternMember, ...]
    min_members_required: int = 2


def _compound(
    name: str,
    molecular_weight: float,
    quantifier_mz: int,
    retention_index: float,
    spectrum: dict[int, float],
    **extra: object,
) -> tuple[str, LibraryCompound]:
    return name, LibraryCompound(
        name=name,
        molecular_weight=molecular_weight,
        quantifier_mz=quantifier_mz,
        retention_index=retention_index,
        spectrum=MappingProxyType(dict(sorted(spectrum.items()))),
        **extra,  # type: ignore[arg-type]
    )


REFERENCE_COMPOUNDS: Mapping[str, LibraryCompound] = MappingProxyType(
    dict(
        [
            # ------------------------------------------------------- styrenics
            _compound(
                "styrene",
                104.0,
                104,
                891.0,
                {104: 100.0, 103: 58.0, 78: 43.0, 51: 27.0, 77: 15.0, 102: 10.0},
                cas_number="100-42-5",
                formula="C8H8",
            ),
            _compound(
                "alpha-methylstyrene",
                118.0,
                118,
                980.0,
                {118: 100.0, 117: 88.0, 103: 34.0, 91: 30.0, 115: 26.0, 78: 15.0},
                cas_number="98-83-9",
            ),
            _compound(
                "2,4-diphenyl-1-butene",
                208.0,
                208,
                1880.0,
                {91: 100.0, 104: 54.0, 117: 30.0, 208: 21.0, 130: 13.0, 193: 9.0},
                cas_number="16606-47-6",
                ri_tolerance=60.0,
            ),
            _compound(
                "2,4,6-triphenyl-1-hexene",
                312.0,
                194,
                2650.0,
                {91: 100.0, 117: 40.0, 104: 35.0, 194: 17.0, 207: 12.0, 312: 4.0},
                ri_tolerance=80.0,
            ),
            _compound(
                "acrylonitrile",
                53.0,
                53,
                530.0,
                {53: 100.0, 52: 70.0, 26: 54.0, 51: 25.0},
                cas_number="107-13-1",
            ),
            _compound(
                "4-vinylcyclohexene",
                108.0,
                54,
                800.0,
                {54: 100.0, 79: 58.0, 108: 36.0, 67: 27.0, 39: 19.0},
                cas_number="100-40-3",
            ),
            # ------------------------------------------------------ polyolefins
            _compound(
                "2,4-dimethyl-1-heptene",
                126.0,
                126,
                894.0,
                {70: 100.0, 41: 38.0, 55: 30.0, 43: 23.0, 56: 18.0, 69: 15.0, 126: 10.0},
                cas_number="19549-87-2",
            ),
            # ------------------------------------------------------------- PET
            _compound(
                "benzoic acid",
                122.0,
                105,
                1170.0,
                {105: 100.0, 122: 80.0, 77: 68.0, 51: 25.0},
                cas_number="65-85-0",
                ri_tolerance=60.0,
            ),
            _compound(
                "vinyl benzoate",
                148.0,
                148,
                1150.0,
                {105: 100.0, 77: 54.0, 148: 18.0, 51: 21.0},
                cas_number="769-78-8",
            ),
            _compound(
                "divinyl terephthalate",
                218.0,
                149,
                1600.0,
                {149: 100.0, 175: 26.0, 121: 21.0, 218: 13.0, 65: 17.0, 76: 12.0},
                ri_tolerance=60.0,
            ),
            _compound(
                "biphenyl",
                154.0,
                154,
                1385.0,
                {154: 100.0, 153: 40.0, 152: 26.0, 76: 12.0},
                cas_number="92-52-4",
            ),
            _compound(
                "acetophenone",
                120.0,
                105,
                1065.0,
                {105: 100.0, 77: 58.0, 120: 27.0, 51: 19.0},
                cas_number="98-86-2",
            ),
            _compound(
                "benzaldehyde",
                106.0,
                105,
                962.0,
                {106: 100.0, 105: 92.0, 77: 84.0, 51: 30.0},
                cas_number="100-52-7",
            ),
            # ------------------------------------------------------- polyamides
            _compound(
                "epsilon-caprolactam",
                113.0,
                113,
                1310.0,
                {113: 100.0, 85: 70.0, 55: 54.0, 56: 50.0, 30: 40.0, 84: 26.0, 42: 21.0},
                cas_number="105-60-2",
                ri_tolerance=60.0,
            ),
            _compound(
                "hexanenitrile",
                97.0,
                97,
                800.0,
                {97: 100.0, 54: 64.0, 41: 56.0, 82: 27.0, 43: 23.0},
                cas_number="628-73-9",
            ),
            _compound(
                "cyclopentanone",
                84.0,
                84,
                800.0,
                {84: 100.0, 55: 88.0, 56: 39.0, 41: 31.0, 39: 21.0},
                cas_number="120-92-3",
            ),
            # -------------------------------------------------------- PVC / PAH
            _compound(
                "benzene",
                78.0,
                78,
                655.0,
                {78: 100.0, 77: 21.0, 51: 21.0, 52: 16.0, 50: 12.0},
                cas_number="71-43-2",
            ),
            _compound(
                "toluene",
                92.0,
                91,
                762.0,
                {91: 100.0, 92: 64.0, 65: 10.0, 39: 8.0},
                cas_number="108-88-3",
            ),
            _compound(
                "indene",
                116.0,
                116,
                1035.0,
                {116: 100.0, 115: 88.0, 89: 9.0, 63: 8.0},
                cas_number="95-13-6",
            ),
            _compound(
                "naphthalene",
                128.0,
                128,
                1181.0,
                {128: 100.0, 127: 15.0, 129: 11.0, 102: 7.0, 51: 6.0},
                cas_number="91-20-3",
            ),
            _compound(
                "chlorobenzene",
                112.0,
                112,
                840.0,
                {112: 100.0, 77: 88.0, 114: 31.0, 51: 21.0},
                cas_number="108-90-7",
            ),
            # -------------------------------------------------------------- PC
            _compound(
                "bisphenol A",
                228.0,
                213,
                2130.0,
                {213: 100.0, 228: 30.0, 119: 21.0, 135: 9.0, 91: 10.0},
                cas_number="80-05-7",
                ri_tolerance=80.0,
            ),
            _compound(
                "phenol",
                94.0,
                94,
                980.0,
                {94: 100.0, 66: 21.0, 65: 15.0, 39: 12.0},
                cas_number="108-95-2",
            ),
            _compound(
                "4-isopropenylphenol",
                134.0,
                134,
                1210.0,
                {134: 100.0, 119: 62.0, 91: 26.0, 77: 15.0},
            ),
            # ------------------------------------------------------------ PMMA
            _compound(
                "methyl methacrylate",
                100.0,
                100,
                620.0,
                {69: 100.0, 100: 40.0, 41: 30.0, 39: 26.0, 59: 12.0},
                cas_number="80-62-6",
            ),
            _compound(
                "methyl methacrylate dimer",
                200.0,
                141,
                1300.0,
                {69: 100.0, 41: 40.0, 141: 19.0, 100: 17.0},
            ),
            # ------------------------------------------------------- additives
            _compound(
                "2,6-di-tert-butyl-4-methylphenol (BHT)",
                220.0,
                205,
                1513.0,
                {205: 100.0, 220: 21.0, 177: 10.0, 145: 8.0, 57: 12.0},
                cas_number="128-37-0",
            ),
            _compound(
                "2,4-di-tert-butylphenol",
                206.0,
                191,
                1513.0,
                {191: 100.0, 206: 29.0, 57: 21.0, 147: 12.0},
                cas_number="96-76-4",
            ),
            _compound(
                "bis(2-ethylhexyl) phthalate (DEHP)",
                390.0,
                149,
                2540.0,
                {149: 100.0, 167: 26.0, 279: 21.0, 57: 15.0, 71: 12.0, 43: 16.0},
                cas_number="117-81-7",
                ri_tolerance=90.0,
            ),
            _compound(
                "dibutyl phthalate (DBP)",
                278.0,
                149,
                1965.0,
                {149: 100.0, 205: 10.0, 223: 8.0, 121: 7.0, 41: 12.0},
                cas_number="84-74-2",
                ri_tolerance=70.0,
            ),
            _compound(
                "octadecanoic acid (stearic acid)",
                284.0,
                60,
                2160.0,
                {60: 100.0, 73: 80.0, 43: 52.0, 129: 30.0, 185: 17.0, 284: 15.0},
                cas_number="57-11-4",
                ri_tolerance=80.0,
            ),
            _compound(
                "erucamide",
                337.0,
                59,
                2700.0,
                {59: 100.0, 72: 54.0, 55: 45.0, 41: 40.0, 69: 27.0},
                cas_number="112-84-5",
                ri_tolerance=100.0,
            ),
            _compound(
                "2-(2H-benzotriazol-2-yl)-p-cresol fragment",
                225.0,
                225,
                2050.0,
                {225: 100.0, 120: 42.0, 92: 25.0, 197: 19.0, 65: 15.0},
                ri_tolerance=90.0,
            ),
            _compound(
                "tetrabromobisphenol A fragment",
                250.0,
                250,
                2200.0,
                {250: 100.0, 252: 92.0, 171: 27.0, 143: 19.0, 63: 15.0},
                ri_tolerance=90.0,
            ),
        ]
    )
)
"""Reference compounds, keyed by name."""


MARKER_PATTERNS: tuple[LibraryPattern, ...] = (
    LibraryPattern(
        name="PS styrene triad",
        polymer=PolymerClass.PS,
        description=(
            "Styrene with its dimer and trimer. The ratio is what identifies "
            "polystyrene; styrene alone is released by several unrelated materials."
        ),
        members=(
            LibraryPatternMember("styrene", MarkerRole.MONOMER, 100.0, is_reference=True),
            LibraryPatternMember("2,4-diphenyl-1-butene", MarkerRole.DIMER, 12.0),
            LibraryPatternMember("2,4,6-triphenyl-1-hexene", MarkerRole.TRIMER, 6.0),
            LibraryPatternMember(
                "alpha-methylstyrene", MarkerRole.PYROLYSIS_FRAGMENT, 4.0
            ),
        ),
        min_members_required=2,
    ),
    LibraryPattern(
        name="PET terephthalate markers",
        polymer=PolymerClass.PET,
        description="Benzoic acid with vinyl benzoate and divinyl terephthalate.",
        members=(
            LibraryPatternMember(
                "benzoic acid", MarkerRole.PYROLYSIS_FRAGMENT, 100.0, is_reference=True
            ),
            LibraryPatternMember("vinyl benzoate", MarkerRole.MONOMER, 46.0),
            LibraryPatternMember("divinyl terephthalate", MarkerRole.DIMER, 28.0),
            LibraryPatternMember("biphenyl", MarkerRole.PYROLYSIS_FRAGMENT, 9.0),
        ),
        min_members_required=2,
    ),
    LibraryPattern(
        name="PA6 caprolactam markers",
        polymer=PolymerClass.PA6,
        description="Caprolactam dominates; the nitrile confirms the polyamide.",
        members=(
            LibraryPatternMember(
                "epsilon-caprolactam", MarkerRole.MONOMER, 100.0, is_reference=True
            ),
            LibraryPatternMember("hexanenitrile", MarkerRole.PYROLYSIS_FRAGMENT, 7.5),
        ),
        min_members_required=1,
    ),
    LibraryPattern(
        name="PVC aromatic cascade",
        polymer=PolymerClass.PVC,
        description=(
            "After HCl elimination the polyene backbone cyclises to benzene, "
            "toluene, indene and naphthalene. The cascade, not benzene alone."
        ),
        members=(
            LibraryPatternMember(
                "benzene", MarkerRole.PYROLYSIS_FRAGMENT, 100.0, is_reference=True
            ),
            LibraryPatternMember("toluene", MarkerRole.PYROLYSIS_FRAGMENT, 35.0),
            LibraryPatternMember("naphthalene", MarkerRole.PYROLYSIS_FRAGMENT, 26.0),
            LibraryPatternMember("indene", MarkerRole.PYROLYSIS_FRAGMENT, 18.0),
        ),
        min_members_required=3,
    ),
    LibraryPattern(
        name="PC bisphenol markers",
        polymer=PolymerClass.PC,
        description="Bisphenol A with phenol and isopropenylphenol.",
        members=(
            LibraryPatternMember(
                "bisphenol A", MarkerRole.MONOMER, 100.0, is_reference=True
            ),
            LibraryPatternMember("phenol", MarkerRole.PYROLYSIS_FRAGMENT, 32.0),
            LibraryPatternMember("4-isopropenylphenol", MarkerRole.PYROLYSIS_FRAGMENT, 24.0),
        ),
        min_members_required=2,
    ),
    LibraryPattern(
        name="PMMA methacrylate markers",
        polymer=PolymerClass.PMMA,
        description="PMMA depolymerises almost quantitatively to its monomer.",
        members=(
            LibraryPatternMember(
                "methyl methacrylate", MarkerRole.MONOMER, 100.0, is_reference=True
            ),
            LibraryPatternMember("methyl methacrylate dimer", MarkerRole.DIMER, 8.0),
        ),
        min_members_required=1,
    ),
    LibraryPattern(
        name="PP branched oligomer markers",
        polymer=PolymerClass.PP,
        description=(
            "2,4-Dimethyl-1-heptene, the propene trimer, is the diagnostic PP peak."
        ),
        members=(
            LibraryPatternMember(
                "2,4-dimethyl-1-heptene", MarkerRole.TRIMER, 100.0, is_reference=True
            ),
        ),
        min_members_required=1,
    ),
    LibraryPattern(
        name="SAN acrylonitrile markers",
        polymer=PolymerClass.SAN,
        description="Styrene together with acrylonitrile.",
        members=(
            LibraryPatternMember("styrene", MarkerRole.MONOMER, 100.0, is_reference=True),
            LibraryPatternMember("acrylonitrile", MarkerRole.MONOMER, 42.0),
        ),
        min_members_required=2,
    ),
)
"""Quantitative marker patterns, one or more per polymer."""


RESPONSE_FACTORS: Mapping[PolymerClass, float] = MappingProxyType(
    {
        PolymerClass.PE: 1.0,
        PolymerClass.PE_LD: 1.0,
        PolymerClass.PE_HD: 1.0,
        PolymerClass.PP: 0.95,
        PolymerClass.PS: 1.35,
        PolymerClass.ABS: 1.05,
        PolymerClass.SAN: 1.1,
        PolymerClass.PET: 0.42,
        PolymerClass.PA6: 0.68,
        PolymerClass.PA66: 0.35,
        PolymerClass.PVC: 0.30,
        PolymerClass.PC: 0.55,
        PolymerClass.PMMA: 0.90,
    }
)
"""GC-amenable pyrolysate yield per unit mass, relative to polyethylene.

The one place where a value *is* shared with the simulator, and deliberately so:
this is not spectral identification evidence but a physical property of the
pyrolysis, and the passport's composition percentages are wrong without it. It is
also the number most in need of laboratory calibration before any percentage
leaves the building.
"""
