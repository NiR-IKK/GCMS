"""Relational schema of the polymer marker library.

The central design decision, and the one the whole of Milestone 3 rests on:
**a polymer is not identified by a peak, it is identified by a pattern.** Finding
styrene proves nothing — styrene turns up from polystyrene, from ABS, from SAN,
and from the thermal degradation of several unrelated materials. What identifies
polystyrene is the *quantitative relationship* between styrene, its dimer and its
trimer. The schema therefore makes :class:`MarkerPattern` a first-class object
with members carrying expected abundance ratios and tolerances, rather than
storing a flat list of "diagnostic compounds".

Retention indices, not retention times, anchor the library to chromatography.
Retention time drifts between runs, columns and instruments; the Kováts index
does not. That is what makes a library built on one instrument usable on another.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "Base",
    "Polymer",
    "Compound",
    "SpectrumPeak",
    "RetentionIndexEntry",
    "MarkerPattern",
    "MarkerPatternMember",
    "ResponseFactor",
]


class Base(DeclarativeBase):
    """Declarative base for the library tables."""


class Polymer(Base):
    """A polymer class the library can report on."""

    __tablename__ = "polymer"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(24), unique=True, index=True)
    """ISO 1043-1 abbreviation, matching ``PolymerClass`` values."""
    name: Mapped[str] = mapped_column(String(120))
    is_polyolefin: Mapped[bool] = mapped_column(default=False)

    patterns: Mapped[list[MarkerPattern]] = relationship(
        back_populates="polymer", cascade="all, delete-orphan"
    )
    response_factors: Mapped[list[ResponseFactor]] = relationship(
        back_populates="polymer", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Polymer(code={self.code!r})"


class Compound(Base):
    """A pyrolysis product the library can recognise."""

    __tablename__ = "compound"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    cas_number: Mapped[str | None] = mapped_column(String(16), default=None)
    formula: Mapped[str | None] = mapped_column(String(40), default=None)
    molecular_weight: Mapped[float] = mapped_column(Float)
    quantifier_mz: Mapped[int] = mapped_column(Integer)
    """Ion used for quantification — selective against the matrix, not necessarily
    the base peak."""
    source: Mapped[str] = mapped_column(
        String(200),
        default="literature",
        comment="Provenance of the spectrum. Recorded because a library validated "
        "against its own source proves nothing.",
    )

    peaks: Mapped[list[SpectrumPeak]] = relationship(
        back_populates="compound", cascade="all, delete-orphan", lazy="selectin"
    )
    retention_indices: Mapped[list[RetentionIndexEntry]] = relationship(
        back_populates="compound", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        CheckConstraint("molecular_weight > 0", name="ck_compound_mw_positive"),
        CheckConstraint("quantifier_mz > 0", name="ck_compound_quantifier_positive"),
    )

    def spectrum(self) -> dict[int, float]:
        """The reference spectrum as ``{m/z: relative intensity}``."""
        return {peak.mz: peak.intensity for peak in self.peaks}

    def __repr__(self) -> str:
        return f"Compound(name={self.name!r})"


class SpectrumPeak(Base):
    """One ion of a reference mass spectrum."""

    __tablename__ = "spectrum_peak"

    id: Mapped[int] = mapped_column(primary_key=True)
    compound_id: Mapped[int] = mapped_column(
        ForeignKey("compound.id", ondelete="CASCADE"), index=True
    )
    mz: Mapped[int] = mapped_column(Integer)
    intensity: Mapped[float] = mapped_column(Float)
    """Relative intensity with the base peak at 100."""

    compound: Mapped[Compound] = relationship(back_populates="peaks")

    __table_args__ = (
        UniqueConstraint("compound_id", "mz", name="uq_spectrum_peak_compound_mz"),
        CheckConstraint("intensity > 0", name="ck_spectrum_peak_positive"),
    )


class RetentionIndexEntry(Base):
    """Kováts retention index of a compound on a given stationary phase.

    Phase-specific by necessity: the same compound has a very different index on a
    non-polar dimethylpolysiloxane and on a polar wax column, and mixing them up
    would make retention agreement worse than useless.
    """

    __tablename__ = "retention_index"

    id: Mapped[int] = mapped_column(primary_key=True)
    compound_id: Mapped[int] = mapped_column(
        ForeignKey("compound.id", ondelete="CASCADE"), index=True
    )
    stationary_phase: Mapped[str] = mapped_column(String(80), default="5%-phenyl")
    value: Mapped[float] = mapped_column(Float)
    tolerance: Mapped[float] = mapped_column(Float, default=30.0)
    """Half-width of the acceptance window, in index units."""

    compound: Mapped[Compound] = relationship(back_populates="retention_indices")

    __table_args__ = (
        UniqueConstraint(
            "compound_id", "stationary_phase", name="uq_retention_index_compound_phase"
        ),
        CheckConstraint("value > 0", name="ck_retention_index_positive"),
    )


class MarkerPattern(Base):
    """A quantitative marker pattern that identifies one polymer.

    Identification requires the members to be present *in the expected ratios*.
    That is what separates "polystyrene is present" from "something released
    styrene".
    """

    __tablename__ = "marker_pattern"

    id: Mapped[int] = mapped_column(primary_key=True)
    polymer_id: Mapped[int] = mapped_column(
        ForeignKey("polymer.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[str] = mapped_column(String(400), default="")
    min_members_required: Mapped[int] = mapped_column(
        Integer,
        default=2,
        comment="How many members must be found before the pattern may be reported. "
        "One peak is never enough.",
    )

    polymer: Mapped[Polymer] = relationship(back_populates="patterns")
    members: Mapped[list[MarkerPatternMember]] = relationship(
        back_populates="pattern", cascade="all, delete-orphan", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"MarkerPattern(name={self.name!r})"


class MarkerPatternMember(Base):
    """One compound inside a marker pattern, with its expected share."""

    __tablename__ = "marker_pattern_member"

    id: Mapped[int] = mapped_column(primary_key=True)
    pattern_id: Mapped[int] = mapped_column(
        ForeignKey("marker_pattern.id", ondelete="CASCADE"), index=True
    )
    compound_id: Mapped[int] = mapped_column(ForeignKey("compound.id"), index=True)
    role: Mapped[str] = mapped_column(String(40))
    """``MarkerRole`` value — monomer, dimer, trimer, and so on."""
    relative_abundance: Mapped[float] = mapped_column(Float)
    """Expected area relative to the pattern's reference member (100 = reference)."""
    tolerance_factor: Mapped[float] = mapped_column(
        Float,
        default=2.5,
        comment="Multiplicative tolerance on the ratio. Pyrolysis ratios vary with "
        "temperature and residence time, so the window is generous by design; "
        "2.5 means a factor of 2.5 either way.",
    )
    is_reference: Mapped[bool] = mapped_column(default=False)

    pattern: Mapped[MarkerPattern] = relationship(back_populates="members")
    compound: Mapped[Compound] = relationship(lazy="selectin")

    __table_args__ = (
        UniqueConstraint(
            "pattern_id", "compound_id", name="uq_marker_member_pattern_compound"
        ),
        CheckConstraint("relative_abundance > 0", name="ck_marker_member_positive"),
        CheckConstraint("tolerance_factor > 1", name="ck_marker_member_tolerance"),
    )


class ResponseFactor(Base):
    """GC-amenable pyrolysate yield per unit mass, relative to polyethylene.

    Persisted because it is the difference between a signal share and a mass
    share, and every composition percentage in the recyclate passport depends on
    it. Pyrolysis temperature is part of the key: the yield of a polymer that
    partly decomposes to non-eluting fragments changes with it.
    """

    __tablename__ = "response_factor"

    id: Mapped[int] = mapped_column(primary_key=True)
    polymer_id: Mapped[int] = mapped_column(
        ForeignKey("polymer.id", ondelete="CASCADE"), index=True
    )
    pyrolysis_temperature_c: Mapped[float] = mapped_column(Float, default=600.0)
    factor: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(200), default="literature estimate")
    measured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    polymer: Mapped[Polymer] = relationship(back_populates="response_factors")

    __table_args__ = (
        UniqueConstraint(
            "polymer_id",
            "pyrolysis_temperature_c",
            name="uq_response_factor_polymer_temperature",
        ),
        CheckConstraint("factor > 0", name="ck_response_factor_positive"),
    )
