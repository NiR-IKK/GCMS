"""Rendering a recyclate passport as HTML and PDF.

The document a customer reads is where a caveat either survives or gets lost, so
the template puts the calibration status and the unassessable regulatory findings
*above* the numbers they qualify rather than in a footnote. A reader who takes
only the headline should still come away with the right impression.

Output is deterministic: the same passport renders byte-identically twice, so two
runs of one sample can be diffed. The issue timestamp is the single varying
element and is taken from the passport itself, not from the clock at render time.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from data_schemas.passport import CalibrationStatus, RecyclatePassport
from pyrecycle_analytics.exceptions import MissingDependencyError

__all__ = ["render_html", "render_pdf"]

_STYLE = """
body { font-family: 'DejaVu Sans', Helvetica, Arial, sans-serif; font-size: 10.5pt;
       color: #1a1a1a; margin: 2.2cm 1.8cm; line-height: 1.45; }
h1 { font-size: 19pt; margin: 0 0 0.1cm; }
h2 { font-size: 13pt; margin: 0.85cm 0 0.25cm; border-bottom: 1.5px solid #444;
     padding-bottom: 0.08cm; }
.sub { color: #555; font-size: 9.5pt; margin-bottom: 0.5cm; }
table { width: 100%; border-collapse: collapse; margin-bottom: 0.35cm; }
th { text-align: left; font-size: 8.5pt; text-transform: uppercase;
     letter-spacing: 0.04em; color: #555; border-bottom: 1px solid #bbb;
     padding: 0.12cm 0.2cm 0.12cm 0; }
td { padding: 0.14cm 0.2cm 0.14cm 0; border-bottom: 1px solid #eee;
     vertical-align: top; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.banner { border-left: 4px solid #b8860b; background: #fdf7e8; padding: 0.3cm 0.4cm;
          margin-bottom: 0.5cm; font-size: 9.5pt; }
.banner.warn { border-left-color: #a33; background: #fdeeee; }
.banner strong { display: block; margin-bottom: 0.1cm; }
.tag { font-size: 8pt; padding: 0.04cm 0.18cm; border-radius: 0.08cm;
       background: #eee; color: #333; }
.tag.high { background: #dff0d8; } .tag.medium { background: #fcf4dd; }
.tag.low, .tag.indicative { background: #f6e0e0; }
.note { font-size: 8.5pt; color: #666; margin: 0.15cm 0 0.4cm; }
.footer { margin-top: 1cm; font-size: 8pt; color: #777;
          border-top: 1px solid #ddd; padding-top: 0.2cm; }
"""


def _rows(rows: list[str]) -> str:
    return "\n".join(rows)


def _polymer_section(passport: RecyclatePassport) -> str:
    if not passport.polymer_fractions:
        return "<p>No polymer could be identified in this sample.</p>"

    rows = [
        f"<tr><td>{escape(str(fraction.polymer))}</td>"
        f"<td class='num'>{fraction.share_percent:.2f}</td>"
        f"<td><span class='tag {fraction.confidence}'>"
        f"{escape(str(fraction.confidence))}</span></td>"
        f"<td>{escape(fraction.marker_pattern)}</td>"
        f"<td class='num'>{len(fraction.markers_found)}/{fraction.markers_expected}</td>"
        "</tr>"
        for fraction in passport.polymer_fractions
    ]
    rows.append(
        f"<tr><td><em>not attributed</em></td>"
        f"<td class='num'>{passport.unassigned_share_percent:.2f}</td>"
        "<td></td><td><em>signal no identified polymer accounts for</em></td>"
        "<td></td></tr>"
    )
    return (
        "<table><tr><th>Polymer</th><th style='text-align:right'>Share %</th>"
        "<th>Confidence</th><th>Marker pattern</th>"
        "<th style='text-align:right'>Markers</th></tr>"
        f"{_rows(rows)}</table>"
        "<p class='note'>Unattributed signal is reported rather than distributed "
        "over the identified polymers; distributing it would inflate every figure "
        "above in proportion to what the method failed to explain.</p>"
    )


def _regulatory_section(passport: RecyclatePassport) -> str:
    rows = []
    for finding in passport.regulatory_findings:
        if finding.exceeds_limit is None:
            verdict = "not assessable" if finding.detected else "—"
        else:
            verdict = "EXCEEDS LIMIT" if finding.exceeds_limit else "within limit"
        rows.append(
            f"<tr><td>{escape(finding.substance)}</td>"
            f"<td>{escape(finding.cas_number or '—')}</td>"
            f"<td>{'detected' if finding.detected else 'not detected'}</td>"
            f"<td class='num'>"
            f"{finding.limit_percent:.2f}" if finding.limit_percent else "<td class='num'>—"
        )
        rows[-1] += f"</td><td>{escape(verdict)}</td></tr>"

    banner = ""
    if passport.unassessable_findings:
        listed = ", ".join(escape(name) for name in passport.unassessable_findings)
        banner = (
            "<div class='banner warn'><strong>Detected, but compliance not "
            "assessable</strong>"
            f"{listed}. Regulatory limits are mass fractions. This method "
            "establishes that a substance is present; quantifying it against a "
            "limit additionally requires calibration standards, which were not "
            "run. No statement about compliance is made.</div>"
        )

    return (
        banner
        + "<table><tr><th>Substance</th><th>CAS</th><th>Result</th>"
        "<th style='text-align:right'>Limit %</th><th>Assessment</th></tr>"
        f"{_rows(rows)}</table>"
    )


def _degradation_section(passport: RecyclatePassport) -> str:
    degradation = passport.degradation
    if degradation is None:
        return "<p>Degradation indices were not computed for this sample.</p>"

    rows = [
        ("Carbonyl index", f"{degradation.carbonyl_index:.4f}"),
        ("Acid share of oxidation products", f"{degradation.acid_share:.3f}"),
        ("Alkene / alkane", f"{degradation.alkene_to_alkane:.3f}"),
        ("Branching index", f"{degradation.branching_index:.3f}"),
    ]
    if degradation.mean_chain_length is not None:
        rows.append(("Mean chain length (comb position)", f"{degradation.mean_chain_length:.2f}"))

    body = _rows(
        [f"<tr><td>{escape(label)}</td><td class='num'>{value}</td></tr>" for label, value in rows]
    )

    if degradation.is_interpretable:
        relative = _rows(
            [
                f"<tr><td>{escape(key.replace('_', ' '))}</td>"
                f"<td class='num'>{value:.2f} ×</td></tr>"
                for key, value in sorted(degradation.relative_to_virgin.items())
            ]
        )
        comparison = (
            "<p class='note'>Relative to the virgin reference "
            f"{escape(degradation.reference_sample_id or 'supplied')}:</p>"
            f"<table>{relative}</table>"
        )
    else:
        comparison = (
            "<div class='banner'><strong>No virgin reference</strong>"
            "These indices are ratios whose absolute level depends on the polymer "
            "and on the pyrolysis temperature. Without a virgin reference of the "
            "same material measured under the same method they cannot be "
            "interpreted, and must not be compared across polymers.</div>"
        )

    return f"<table>{body}</table>{comparison}"


def render_html(passport: RecyclatePassport) -> str:
    """Render a passport as a self-contained HTML document.

    Args:
        passport: The passport to render.

    Returns:
        Complete HTML, with all styling inlined.
    """
    status = passport.calibration_status
    banner_class = "banner" if status is CalibrationStatus.CALIBRATED else "banner warn"
    provenance = passport.provenance

    provenance_rows = _rows(
        [
            f"<tr><td>{escape(label)}</td><td>{escape(str(value))}</td></tr>"
            for label, value in [
                ("Source file", provenance.source_file or "—"),
                ("Checksum (SHA-256)", (provenance.source_checksum or "—")[:32]),
                ("Preprocessing", " → ".join(provenance.preprocessing_steps) or "—"),
                ("Matrix subtracted", "yes" if provenance.matrix_subtracted else "no"),
                (
                    "Matrix share of signal",
                    f"{provenance.matrix_explained_fraction:.1%}",
                ),
                ("Components resolved", provenance.n_components_resolved),
                ("Compounds identified", provenance.n_compounds_identified),
                ("Retention-index anchor", provenance.retention_index_anchor or "none"),
                ("Library", provenance.library_version),
                ("Software", provenance.software_version),
            ]
        ]
    )
    warnings = _rows(
        [f"<li>{escape(warning)}</li>" for warning in provenance.warnings]
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Recyclate passport — {escape(passport.sample.sample_id)}</title>
<style>{_STYLE}</style></head><body>
<h1>Digital recyclate passport</h1>
<div class="sub">{escape(passport.sample.sample_id)}
 · {escape(str(passport.sample.stream))}
 · issued {passport.issued_at.strftime("%Y-%m-%d %H:%M UTC")}
 · schema {escape(passport.schema_version)}</div>

<div class="{banner_class}"><strong>Calibration status: {escape(str(status))}</strong>
{escape(status.disclaimer)}</div>

<h2>Polymer composition</h2>
{_polymer_section(passport)}

<h2>Regulated substances</h2>
{_regulatory_section(passport)}

<h2>Degradation</h2>
{_degradation_section(passport)}

<h2>Provenance</h2>
<table>{provenance_rows}</table>

<h2>Notes and limitations</h2>
<ul class="note">{warnings}</ul>

<div class="footer">Generated by PyRecycle-Analytics
{escape(provenance.software_version)}. Analytical method: pyrolysis GC/MS at
{passport.acquisition.pyrolysis.temperature_c:.0f} °C.</div>
</body></html>"""


def render_pdf(passport: RecyclatePassport, path: str | Path) -> Path:
    """Render a passport to PDF.

    Args:
        passport: The passport to render.
        path: Destination; ``.pdf`` is appended when missing.

    Returns:
        The path written.

    Raises:
        MissingDependencyError: If WeasyPrint is not installed.
    """
    try:
        from weasyprint import HTML  # noqa: PLC0415 - optional dependency
    except ImportError as error:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            package="weasyprint",
            purpose="rendering the recyclate passport as PDF",
            extra="report",
        ) from error

    path = Path(path)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=render_html(passport)).write_pdf(str(path))
    return path
