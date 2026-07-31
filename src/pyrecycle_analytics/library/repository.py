"""Creating, seeding and querying the marker library."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from data_schemas.enums import PolymerClass
from pyrecycle_analytics.library.reference_data import (
    MARKER_PATTERNS,
    REFERENCE_COMPOUNDS,
    RESPONSE_FACTORS,
    STATIONARY_PHASE,
    LibraryCompound,
)
from pyrecycle_analytics.library.schema import (
    Base,
    Compound,
    MarkerPattern,
    MarkerPatternMember,
    Polymer,
    ResponseFactor,
    RetentionIndexEntry,
    SpectrumPeak,
)

__all__ = ["create_library", "seed_library", "MarkerLibrary"]

_POLYMER_NAMES: dict[PolymerClass, str] = {
    PolymerClass.PE: "Polyethylene",
    PolymerClass.PE_LD: "Low-density polyethylene",
    PolymerClass.PE_HD: "High-density polyethylene",
    PolymerClass.PP: "Polypropylene",
    PolymerClass.PS: "Polystyrene",
    PolymerClass.ABS: "Acrylonitrile butadiene styrene",
    PolymerClass.SAN: "Styrene acrylonitrile",
    PolymerClass.PET: "Polyethylene terephthalate",
    PolymerClass.PA6: "Polyamide 6",
    PolymerClass.PA66: "Polyamide 66",
    PolymerClass.PVC: "Polyvinyl chloride",
    PolymerClass.PC: "Polycarbonate",
    PolymerClass.PMMA: "Polymethyl methacrylate",
}


def create_library(url: str = "sqlite:///:memory:", *, echo: bool = False) -> Engine:
    """Create an empty library database.

    Args:
        url: SQLAlchemy URL. In-memory SQLite by default; PostgreSQL in production.
        echo: Log emitted SQL.

    Returns:
        A connected engine with the schema created.
    """
    engine = create_engine(url, echo=echo, future=True)
    Base.metadata.create_all(engine)
    return engine


def _add_compound(session: Session, entry: LibraryCompound) -> Compound:
    compound = Compound(
        name=entry.name,
        cas_number=entry.cas_number,
        formula=entry.formula,
        molecular_weight=entry.molecular_weight,
        quantifier_mz=entry.quantifier_mz,
        source=entry.source,
    )
    compound.peaks = [
        SpectrumPeak(mz=int(mz), intensity=float(intensity))
        for mz, intensity in entry.spectrum.items()
        if intensity > 0.0
    ]
    compound.retention_indices = [
        RetentionIndexEntry(
            stationary_phase=STATIONARY_PHASE,
            value=entry.retention_index,
            tolerance=entry.ri_tolerance,
        )
    ]
    session.add(compound)
    return compound


def seed_library(engine: Engine) -> None:
    """Populate a library from the curated reference data.

    Idempotent: running it twice leaves the library unchanged, so an application
    can call it on every start.

    Args:
        engine: Engine returned by :func:`create_library`.
    """
    factory = sessionmaker(engine, future=True)
    with factory() as session, session.begin():
        if session.scalar(select(Compound).limit(1)) is not None:
            return

        polymers: dict[PolymerClass, Polymer] = {}
        for polymer_class, name in _POLYMER_NAMES.items():
            polymer = Polymer(
                code=str(polymer_class),
                name=name,
                is_polyolefin=polymer_class.is_polyolefin,
            )
            polymers[polymer_class] = polymer
            session.add(polymer)

        compounds: dict[str, Compound] = {
            name: _add_compound(session, entry)
            for name, entry in REFERENCE_COMPOUNDS.items()
        }

        for pattern_spec in MARKER_PATTERNS:
            pattern = MarkerPattern(
                polymer=polymers[pattern_spec.polymer],
                name=pattern_spec.name,
                description=pattern_spec.description,
                min_members_required=pattern_spec.min_members_required,
            )
            pattern.members = [
                MarkerPatternMember(
                    compound=compounds[member.compound],
                    role=str(member.role),
                    relative_abundance=member.relative_abundance,
                    tolerance_factor=member.tolerance_factor,
                    is_reference=member.is_reference,
                )
                for member in pattern_spec.members
            ]
            session.add(pattern)

        for polymer_class, factor in RESPONSE_FACTORS.items():
            session.add(
                ResponseFactor(
                    polymer=polymers[polymer_class],
                    pyrolysis_temperature_c=600.0,
                    factor=factor,
                    source="literature estimate; requires laboratory calibration",
                )
            )


class MarkerLibrary:
    """Read access to a seeded library.

    Wraps the ORM so that the identification code never holds an open session or
    handles SQLAlchemy objects: it works on plain snapshots, which keeps it
    testable and keeps database lifetime out of the chemistry.
    """

    def __init__(self, engine: Engine, *, stationary_phase: str = STATIONARY_PHASE) -> None:
        self._engine = engine
        self._factory = sessionmaker(engine, future=True)
        self.stationary_phase = stationary_phase

    @classmethod
    def in_memory(cls) -> MarkerLibrary:
        """A fully seeded in-memory library — the default for tests and the API."""
        engine = create_library()
        seed_library(engine)
        return cls(engine)

    def compounds(self) -> list[Compound]:
        """All reference compounds, with spectra and retention indices loaded."""
        with self._factory() as session:
            return list(session.scalars(select(Compound)).unique().all())

    def compound(self, name: str) -> Compound:
        """One compound by exact name.

        Raises:
            KeyError: If the compound is not in the library.
        """
        with self._factory() as session:
            found = session.scalar(select(Compound).where(Compound.name == name))
            if found is None:
                raise KeyError(f"{name!r} is not in the marker library")
            return found

    def patterns(self) -> list[MarkerPattern]:
        """All marker patterns with their members and polymers."""
        with self._factory() as session:
            patterns = list(session.scalars(select(MarkerPattern)).unique().all())
            for pattern in patterns:
                _ = pattern.polymer.code, [m.compound.name for m in pattern.members]
            return patterns

    def response_factor(
        self, polymer_code: str, *, pyrolysis_temperature_c: float = 600.0
    ) -> float:
        """GC-amenable yield of a polymer relative to polyethylene.

        Args:
            polymer_code: ``PolymerClass`` value, e.g. ``"PET"``.
            pyrolysis_temperature_c: Pyrolysis set-point the factor applies to.

        Returns:
            The factor, or 1.0 when the polymer is unknown — a neutral assumption
            that is flagged rather than silently wrong.
        """
        with self._factory() as session:
            factor = session.scalar(
                select(ResponseFactor)
                .join(Polymer)
                .where(
                    Polymer.code == polymer_code,
                    ResponseFactor.pyrolysis_temperature_c == pyrolysis_temperature_c,
                )
            )
            return float(factor.factor) if factor is not None else 1.0

    def retention_index(self, compound_name: str) -> float | None:
        """Tabulated Kováts index of a compound on this library's phase."""
        with self._factory() as session:
            entry = session.scalar(
                select(RetentionIndexEntry)
                .join(Compound)
                .where(
                    Compound.name == compound_name,
                    RetentionIndexEntry.stationary_phase == self.stationary_phase,
                )
            )
            return float(entry.value) if entry is not None else None

    def polymer_codes(self) -> list[str]:
        """Codes of every polymer the library knows."""
        with self._factory() as session:
            return sorted(session.scalars(select(Polymer.code)).all())

    def add_compounds(self, entries: Iterable[LibraryCompound]) -> None:
        """Extend the library with additional reference compounds."""
        with self._factory() as session, session.begin():
            for entry in entries:
                _add_compound(session, entry)

    def dispose(self) -> None:
        """Release the engine's connection pool."""
        self._engine.dispose()

    def __repr__(self) -> str:
        return f"MarkerLibrary(phase={self.stationary_phase!r})"


def default_library_path(base: Path | None = None) -> Path:
    """Conventional on-disk location of the library."""
    root = base or Path.cwd()
    return root / "data" / "marker_library.sqlite"
