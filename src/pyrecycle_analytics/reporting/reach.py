"""Regulated substances that Py-GC/MS can look for.

The table maps a regulated substance to the pyrolysis marker that betrays it,
along with the legal basis and the threshold. Two properties of this list matter
more than its length:

* **Only substances that actually reach the detector are listed.** Heavy-metal
  stabilisers (cadmium, lead) are regulated in plastics and are *not* here,
  because they do not elute from a GC column. Listing them with a permanent "not
  detected" would be worse than omitting them: a reader would take the absence of
  a finding as evidence of absence.
* **A threshold in the table is not a promise that it can be tested against.**
  The limits are mass fractions. Without calibration standards this platform
  establishes presence, not concentration, and the passport says so per finding.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RegulatedSubstance", "REACH_WATCHLIST", "NON_GC_AMENABLE_NOTE"]


@dataclass(frozen=True, slots=True)
class RegulatedSubstance:
    """A regulated substance and the marker that reveals it.

    Attributes:
        name: Substance name as it should appear in the report.
        cas_number: CAS registry number.
        regulation: Legal basis.
        limit_percent: Threshold as a mass fraction, where one exists.
        marker_compound: Library compound whose detection implies this substance.
        note: Interpretation caveat carried into the passport.
    """

    name: str
    cas_number: str | None
    regulation: str
    limit_percent: float | None
    marker_compound: str
    note: str = ""


REACH_WATCHLIST: tuple[RegulatedSubstance, ...] = (
    RegulatedSubstance(
        name="Bis(2-ethylhexyl) phthalate (DEHP)",
        cas_number="117-81-7",
        regulation="REACH Annex XIV / Annex XVII entry 51",
        limit_percent=0.1,
        marker_compound="bis(2-ethylhexyl) phthalate (DEHP)",
        note="Plasticiser; SVHC, reprotoxic. Detected via its m/z 149 phthalate ion.",
    ),
    RegulatedSubstance(
        name="Dibutyl phthalate (DBP)",
        cas_number="84-74-2",
        regulation="REACH Annex XIV / Annex XVII entry 51",
        limit_percent=0.1,
        marker_compound="dibutyl phthalate (DBP)",
        note=(
            "Shares the m/z 149 quantifier with every other ortho-phthalate; the "
            "assignment rests on retention, so a retention-index anchor matters here."
        ),
    ),
    RegulatedSubstance(
        name="Bisphenol A",
        cas_number="80-05-7",
        regulation="REACH Annex XVII entry 66 / SVHC list",
        limit_percent=None,
        marker_compound="bisphenol A",
        note=(
            "In a recyclate it usually indicates a polycarbonate fraction rather "
            "than free monomer; the passport reports the polymer finding alongside."
        ),
    ),
    RegulatedSubstance(
        name="Tetrabromobisphenol A (TBBPA)",
        cas_number="79-94-7",
        regulation="POP Regulation (EU) 2019/1021 context; SVHC assessment",
        limit_percent=None,
        marker_compound="tetrabromobisphenol A fragment",
        note=(
            "Brominated flame retardant, typical of WEEE fractions. Detected as a "
            "brominated fragment; the bromine isotope pattern is the confirmation."
        ),
    ),
    RegulatedSubstance(
        name="Naphthalene",
        cas_number="91-20-3",
        regulation="REACH Annex XVII entry 28 (carcinogen category 2)",
        limit_percent=0.1,
        marker_compound="naphthalene",
        note=(
            "In a pyrogram naphthalene is normally a PVC pyrolysis product rather "
            "than an ingredient; read it together with the PVC finding."
        ),
    ),
    RegulatedSubstance(
        name="Benzene",
        cas_number="71-43-2",
        regulation="REACH Annex XVII entry 5",
        limit_percent=0.1,
        marker_compound="benzene",
        note=(
            "Almost always formed during pyrolysis, not present in the material. "
            "A detection here is evidence about the pyrolysate, not the sample."
        ),
    ),
)
"""Regulated substances with a usable Py-GC/MS marker."""


NON_GC_AMENABLE_NOTE = (
    "Heavy-metal stabilisers (cadmium, lead, chromium VI), short-chain chlorinated "
    "paraffins and inorganic flame retardants are regulated in plastics but do not "
    "elute from a GC column. They are outside the scope of this method and are "
    "deliberately not listed as 'not detected'; establishing their absence requires "
    "XRF or ICP-MS."
)
"""Statement that must appear in every passport's regulatory section.

Omitting it would let a reader take the passport's silence as a clean bill of
health for substances the method never looked for.
"""
