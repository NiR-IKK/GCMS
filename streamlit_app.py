"""PyRecycle-Analytics — Streamlit-Oberfläche.

Zweck: eigene Py-GC/MS-Messungen einlesen, sichten, vorverarbeiten und
auswerten. Die Datei wird per Upload entgegengenommen (.CDF / .mzML / .mzXML) —
die synthetischen Rezepturen stehen daneben nur als Demo bereit, damit die
Oberfläche auch ohne Messdaten ausprobiert werden kann.

Was diese Oberfläche leistet:

* Rohdaten-Import inklusive Formaterkennung, Provenienz und Reader-Warnungen
* TIC, Ionenspuren (EIC) und Massenspektren an beliebiger Retentionszeit
* Baseline-Korrektur und Glättung mit direktem Vorher/Nachher-Vergleich
* die vollständige Auswertung bis zum Rezyklat-Pass, mit Download als HTML/JSON
* Export des vorverarbeiteten Laufs

Die Ionenspur-Ansicht bleibt bewusst ein Sichtwerkzeug: eine Ionenspur ist kein
Identitätsnachweis. Was ein belastbares Ergebnis ist, steht im Reiter
*Auswertung* — samt Kalibrierstatus und den Einschränkungen, die dazugehören.

Start::

    pip install -e '.[all]'
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pyrecycle_analytics import PyRecycleError, __version__
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.ingestion import (
    SUPPORTED_EXTENSIONS,
    pyopenms_available,
    read_pyrogram_bytes,
)
from pyrecycle_analytics.preprocessing import PreprocessingConfig, estimate_noise_sigma, preprocess

# Ionen, die beim Sichten eines Rezyklat-Pyrogramms erfahrungsgemäß zuerst
# interessieren. Bewusst nur eine Anzeigehilfe: eine Ionenspur ist kein
# Identitätsnachweis. Die quantitative Marker-Logik (Triaden-Verhältnisse,
# Degradations-Index) steckt im Reiter "Auswertung".
ION_PRESETS: dict[str, tuple[float, ...]] = {
    "Polyolefin-Matrix – Alkane (43, 57, 71)": (43.0, 57.0, 71.0),
    "Polyolefin-Matrix – Alkene (41, 55, 69)": (41.0, 55.0, 69.0),
    "Verzweigung / PP – iso-Alkene (56, 70, 84)": (56.0, 70.0, 84.0),
    "PP-Trimer 2,4-Dimethyl-1-hepten (126)": (126.0,),
    "PS – Styrol (104, 103, 78)": (104.0, 103.0, 78.0),
    "PS – Dimer/Trimer (91, 117, 194)": (91.0, 117.0, 194.0),
    "PET – Benzoesäure / Terephthalat (105, 149)": (105.0, 149.0),
    "PA6 – Caprolactam (113, 85, 55)": (113.0, 85.0, 55.0),
    "PVC – Aromaten (78, 91, 128)": (78.0, 91.0, 128.0),
    "Oxidation – Carbonyle (44, 58, 60, 73)": (44.0, 58.0, 60.0, 73.0),
    "Weichmacher – Phthalate (149, 167, 279)": (149.0, 167.0, 279.0),
    "Säulenbluten – Siloxane (73, 147, 207)": (73.0, 147.0, 207.0),
}

# Obergrenze für gezeichnete Punkte; darüber wird zum Anzeigen ausgedünnt.
# Bewusst ein Kommentar und kein Attribut-Docstring: Streamlits "magic" rendert
# jeden freistehenden String-Ausdruck auf Modulebene als Seiteninhalt.
MAX_PLOT_POINTS = 4000


# ---------------------------------------------------------------------------
# Datenbeschaffung
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Rohdaten werden eingelesen …")
def _load_upload(
    data: bytes,
    filename: str,
    mz_range: tuple[float, float] | None,
    rt_range_s: tuple[float, float] | None,
) -> PyrogramDataCube:
    """Eine hochgeladene Messdatei einlesen (gecacht über den Dateiinhalt)."""
    return read_pyrogram_bytes(
        data, filename, mz_range=mz_range, rt_range_s=rt_range_s
    )


@st.cache_data(show_spinner="Demo-Pyrogramm wird simuliert …")
def _load_demo(recipe_name: str, seed: int) -> tuple[PyrogramDataCube, dict[str, float]]:
    """Ein synthetisches Pyrogramm als Demo erzeugen.

    Der Generator liegt in ``tests/``, weil er Prüfmittel und nicht Produkt ist.
    Für die Demo wird er von hier aus genutzt; falls das Testpaket nicht
    mitinstalliert wurde, meldet die Oberfläche das sauber zurück.
    """
    from tests.synthetic_data import RECIPES, SyntheticPyrogramGenerator

    sample = SyntheticPyrogramGenerator(seed=seed).generate(RECIPES[recipe_name])
    return sample.cube, dict(sample.truth.blend_fractions)


def _demo_recipe_names() -> list[str]:
    """Namen der Demo-Rezepturen, leer wenn das Testpaket fehlt."""
    try:
        from tests.synthetic_data import RECIPES
    except ImportError:
        return []
    return sorted(RECIPES)


# ---------------------------------------------------------------------------
# Darstellung
# ---------------------------------------------------------------------------


def _thin(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Für die Anzeige ausdünnen, ohne Peakmaxima zu verlieren.

    Ein 60-Minuten-Lauf bei 20 Hz hat 72 000 Punkte. Naives Slicing würde genau
    die schmalen Peaks verschlucken, auf die es ankommt, deshalb wird je Block
    das Maximum behalten.
    """
    if x.size <= MAX_PLOT_POINTS:
        return x, y
    block = int(np.ceil(x.size / MAX_PLOT_POINTS))
    usable = (x.size // block) * block
    x_blocks = x[:usable].reshape(-1, block)
    y_blocks = y[:usable].reshape(-1, block)
    peak_positions = y_blocks.argmax(axis=1)
    rows = np.arange(x_blocks.shape[0])
    return x_blocks[rows, peak_positions], y_blocks[rows, peak_positions]


def _chromatogram_figure(
    traces: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    y_label: str = "Intensität",
) -> go.Figure:
    """Liniendiagramm über die Retentionszeit."""
    figure = go.Figure()
    for name, (x, y) in traces.items():
        thin_x, thin_y = _thin(x, y)
        figure.add_trace(
            go.Scatter(x=thin_x / 60.0, y=thin_y, mode="lines", name=name, line={"width": 1.2})
        )
    figure.update_layout(
        title=title,
        xaxis_title="Retentionszeit [min]",
        yaxis_title=y_label,
        height=420,
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        hovermode="x unified",
    )
    return figure


def _spectrum_figure(mz_axis: np.ndarray, intensities: np.ndarray, title: str) -> go.Figure:
    """Strichspektrum; leere Kanäle werden nicht gezeichnet."""
    visible = intensities > 0.0
    figure = go.Figure(
        go.Bar(x=mz_axis[visible], y=intensities[visible], width=0.6, name="Intensität")
    )
    figure.update_layout(
        title=title,
        xaxis_title="m/z",
        yaxis_title="Intensität",
        height=360,
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
        bargap=0.0,
    )
    return figure


def _metadata_table(cube: PyrogramDataCube) -> pd.DataFrame:
    """Kennzahlen des Laufs als Tabelle.

    Alle Werte sind Text. Eine Spalte mit gemischten Typen wird zu ``object``,
    und Streamlit reicht die an Arrow weiter, das sie ablehnt — die Tabelle
    verschwände dann hinter einer Serialisierungsmeldung.
    """
    rt_start, rt_end = cube.rt_range_s
    metadata = cube.metadata
    rows: list[tuple[str, str]] = [
        ("Probe", metadata.sample.sample_id),
        ("Quelle", str(metadata.source_path or "—")),
        ("Format", str(metadata.source_format)),
        ("Scans", f"{cube.n_scans:,}".replace(",", " ")),
        ("m/z-Kanäle", str(cube.n_mz)),
        ("m/z-Bereich", f"{cube.mz_axis[0]:.0f} – {cube.mz_axis[-1]:.0f}"),
        ("Retentionsbereich", f"{rt_start / 60.0:.2f} – {rt_end / 60.0:.2f} min"),
        ("Scanrate", f"{1.0 / cube.mean_scan_period_s:.2f} Hz" if cube.mean_scan_period_s else "—"),
        ("Gesamtsignal", f"{cube.total_signal:.4g}"),
        ("SHA-256", (metadata.source_checksum or "—")[:16]),
    ]
    return pd.DataFrame(rows, columns=["Kennwert", "Wert"])


# ---------------------------------------------------------------------------
# Seitenaufbau
# ---------------------------------------------------------------------------


def _sidebar_source() -> tuple[PyrogramDataCube | None, dict[str, float] | None]:
    """Datenquelle wählen und laden. Gibt Cube und optionale Soll-Zusammensetzung zurück."""
    st.sidebar.header("Datenquelle")

    demo_recipes = _demo_recipe_names()
    modes = ["Eigene Messung hochladen"]
    if demo_recipes:
        modes.append("Demo (simuliert)")
    mode = st.sidebar.radio("Woher kommen die Daten?", modes, label_visibility="collapsed")

    st.sidebar.caption(
        "Import-Fenster: begrenzt Speicherbedarf und schneidet Lösemittelfront "
        "und Bluten-Rampe ab."
    )
    use_mz = st.sidebar.checkbox("m/z-Bereich begrenzen", value=False)
    mz_range = None
    if use_mz:
        mz_low = st.sidebar.number_input("m/z von", value=35.0, min_value=1.0, step=1.0)
        mz_high = st.sidebar.number_input("m/z bis", value=550.0, min_value=2.0, step=1.0)
        mz_range = (float(mz_low), float(mz_high))

    use_rt = st.sidebar.checkbox("Retentionsfenster begrenzen", value=False)
    rt_range_s = None
    if use_rt:
        rt_low = st.sidebar.number_input("RT von [min]", value=1.0, min_value=0.0, step=0.5)
        rt_high = st.sidebar.number_input("RT bis [min]", value=60.0, min_value=0.1, step=0.5)
        rt_range_s = (float(rt_low) * 60.0, float(rt_high) * 60.0)

    if mode == "Demo (simuliert)":
        recipe = st.sidebar.selectbox("Rezeptur", demo_recipes, index=demo_recipes.index(
            "pcr_mixed_polyolefin"
        ) if "pcr_mixed_polyolefin" in demo_recipes else 0)
        seed = int(st.sidebar.number_input("Seed", value=42, min_value=0, step=1))
        cube, blend = _load_demo(recipe, seed)
        if rt_range_s is not None or mz_range is not None:
            try:
                cube = cube.window(rt_range_s=rt_range_s, mz_range=mz_range)
            except ValueError as error:
                st.sidebar.error(f"Import-Fenster passt nicht zum Lauf: {error}")
                return None, None
        return cube, blend

    extensions = sorted({extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS})
    upload = st.sidebar.file_uploader(
        "Messdatei", type=extensions, help="ANDI-MS/AIA netCDF (.CDF) oder mzML / mzXML"
    )
    if upload is None:
        return None, None

    try:
        cube = _load_upload(upload.getvalue(), upload.name, mz_range, rt_range_s)
    except PyRecycleError as error:
        st.sidebar.error(str(error))
        return None, None
    return cube, None


def _sidebar_preprocessing() -> PreprocessingConfig | None:
    """Vorverarbeitungskette konfigurieren; ``None`` bedeutet 'aus'."""
    st.sidebar.header("Vorverarbeitung")
    if not st.sidebar.checkbox("Aktivieren", value=True):
        return None

    smoothing = st.sidebar.selectbox("Glättung", ["savgol", "gaussian", "none"], index=0)
    window = 7
    polyorder = 2
    sigma_scans = 1.0
    if smoothing == "savgol":
        window = int(st.sidebar.slider("Fensterbreite [Scans, ungerade]", 3, 31, 7, step=2))
        polyorder = int(st.sidebar.slider("Polynomgrad", 1, 4, 2))
    elif smoothing == "gaussian":
        sigma_scans = float(st.sidebar.slider("Kernbreite σ [Scans]", 0.2, 5.0, 1.0, step=0.1))

    baseline = st.sidebar.selectbox("Baseline", ["asls", "snip", "none"], index=0)
    lam_exponent = 7.0
    asls_p = 0.005
    asls_iterations = 10
    snip_iterations = 40
    if baseline == "asls":
        lam_exponent = float(st.sidebar.slider("Steifigkeit log₁₀(λ)", 4.0, 10.0, 7.0, step=0.5))
        asls_p = float(st.sidebar.slider("Asymmetrie p", 0.001, 0.1, 0.005, step=0.001,
                                         format="%.3f"))
        asls_iterations = int(st.sidebar.slider("Iterationen", 2, 40, 10))
        st.sidebar.caption(
            "Der Restfehler unter einem Peak fällt etwa wie 1/λ. Unter λ ≈ 1e6 wird "
            "messbar Peakfläche abgeschnitten."
        )
    elif baseline == "snip":
        snip_iterations = int(st.sidebar.slider("Fensterbreite [Scans]", 5, 200, 40))

    return PreprocessingConfig(
        smoothing=smoothing,  # type: ignore[arg-type]
        savgol_window=window,
        savgol_polyorder=polyorder,
        gaussian_sigma_scans=sigma_scans,
        baseline=baseline,  # type: ignore[arg-type]
        asls_lam=10.0**lam_exponent,
        asls_p=asls_p,
        asls_iterations=asls_iterations,
        snip_iterations=snip_iterations,
    )


def _tab_overview(
    cube: PyrogramDataCube, blend: dict[str, float] | None
) -> None:
    """Metadaten, Warnungen und Totalionenstrom."""
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Lauf")
        st.dataframe(_metadata_table(cube), hide_index=True, width="stretch")
        if blend is not None:
            st.caption("Simulierte Soll-Zusammensetzung (Massenanteile):")
            st.dataframe(
                pd.DataFrame(
                    sorted(blend.items()), columns=["Polymer", "Massenanteil"]
                ).assign(Massenanteil=lambda frame: frame["Massenanteil"].map("{:.1%}".format)),
                hide_index=True,
                width="stretch",
            )

    with right:
        st.subheader("Totalionenstrom")
        st.plotly_chart(
            _chromatogram_figure({"TIC": (cube.retention_times, cube.tic)}, "TIC"),
            width="stretch",
        )

    if cube.metadata.reader_warnings:
        st.warning("Hinweise des Readers:")
        for warning in cube.metadata.reader_warnings:
            st.markdown(f"- {warning}")

    if cube.metadata.preprocessing:
        st.caption(
            "Verarbeitungskette: "
            + " → ".join(step.name for step in cube.metadata.preprocessing)
        )


def _tab_ion_traces(cube: PyrogramDataCube) -> None:
    """Ionenspuren mit domänentypischen Voreinstellungen."""
    st.subheader("Ionenspuren")
    st.caption(
        "Eine Ionenspur zeigt, wo ein Fragment auftaucht — sie ist kein "
        "Identitätsnachweis. Quantitative Marker-Triaden und Degradations-Kennzahlen "
        "folgen in Meilenstein 3."
    )

    preset = st.selectbox("Voreinstellung", ["(eigene Auswahl)", *ION_PRESETS])
    if preset == "(eigene Auswahl)":
        raw = st.text_input("m/z-Werte, kommagetrennt", value="57, 55, 104")
        try:
            ions = tuple(float(part) for part in raw.split(",") if part.strip())
        except ValueError:
            st.error("Bitte nur Zahlen, durch Komma getrennt.")
            return
    else:
        ions = ION_PRESETS[preset]

    if not ions:
        st.info("Keine m/z-Werte gewählt.")
        return

    available = [mz for mz in ions if abs(cube.mz_axis - mz).min() <= 0.5]
    missing = [mz for mz in ions if mz not in available]
    if missing:
        st.warning(
            "Nicht im aufgenommenen m/z-Bereich und daher ausgelassen: "
            + ", ".join(f"{mz:.0f}" for mz in missing)
        )
    if not available:
        return

    normalise = st.checkbox("Auf jeweiliges Maximum normieren", value=False)
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mz in available:
        trace = cube.eic(mz)
        if normalise and trace.max() > 0:
            trace = trace / trace.max()
        traces[f"m/z {mz:.0f}"] = (cube.retention_times, trace)

    st.plotly_chart(
        _chromatogram_figure(
            traces, preset, "relative Intensität" if normalise else "Intensität"
        ),
        width="stretch",
    )


def _tab_spectrum(cube: PyrogramDataCube) -> None:
    """Massenspektrum an einer wählbaren Retentionszeit."""
    st.subheader("Massenspektrum")
    rt_start, rt_end = cube.rt_range_s

    retention_min = st.slider(
        "Retentionszeit [min]",
        min_value=float(rt_start / 60.0),
        max_value=float(rt_end / 60.0),
        value=float((rt_start + 0.4 * (rt_end - rt_start)) / 60.0),
        step=float(max(cube.mean_scan_period_s / 60.0, 1e-4)),
    )
    retention_s = retention_min * 60.0

    average_width = st.slider("Mittelung über ± [s]", 0.0, 30.0, 2.0, step=0.5)
    subtract_background = st.checkbox(
        "Untergrund abziehen (Fenster daneben)",
        value=False,
        help="Klassischer manueller Abzug eines Blindbereichs — der Vergleichsmaßstab "
        "für die automatische Deconvolution ab Meilenstein 2.",
    )

    if average_width > 0.0:
        background = None
        if subtract_background:
            offset = 6.0 * max(average_width, 1.0)
            background = (
                min(retention_s + offset, rt_end - 1e-6),
                min(retention_s + offset + 2.0 * average_width, rt_end),
            )
        spectrum = cube.averaged_spectrum(
            max(retention_s - average_width, rt_start),
            min(retention_s + average_width, rt_end),
            background_window_s=background,
        )
    else:
        spectrum = cube.mass_spectrum_at(retention_s)

    scan = cube.nearest_scan(retention_s)
    st.plotly_chart(
        _spectrum_figure(
            cube.mz_axis, spectrum, f"Scan {scan} bei {cube.retention_times[scan] / 60.0:.3f} min"
        ),
        width="stretch",
    )

    if spectrum.max() > 0:
        order = np.argsort(spectrum)[::-1][:10]
        st.caption("Stärkste Ionen (auf Basispeak normiert):")
        st.dataframe(
            pd.DataFrame(
                {
                    "m/z": cube.mz_axis[order].astype(int),
                    "rel. Intensität": (100.0 * spectrum[order] / spectrum.max()).round(1),
                }
            ),
            hide_index=True,
            width="stretch",
        )


def _tab_preprocessing(raw: PyrogramDataCube, processed: PyrogramDataCube | None) -> None:
    """Vorher/Nachher-Vergleich der Vorverarbeitung."""
    st.subheader("Vorverarbeitung")
    if processed is None:
        st.info("Vorverarbeitung ist in der Seitenleiste deaktiviert.")
        return

    channel = st.selectbox(
        "Kanal für den Vergleich",
        ["Totalionenstrom", "m/z 207 (Säulenbluten)", "m/z 57 (Alkane)", "m/z 149 (Phthalate)"],
    )
    if channel == "Totalionenstrom":
        before, after = raw.tic, processed.tic
    else:
        mz = float(channel.split()[1])
        before, after = raw.eic(mz), processed.eic(mz)

    st.plotly_chart(
        _chromatogram_figure(
            {
                "vorher": (raw.retention_times, before),
                "nachher": (processed.retention_times, after),
            },
            f"{channel}: vor und nach der Vorverarbeitung",
        ),
        width="stretch",
    )

    left, middle, right = st.columns(3)
    left.metric("Signal vorher", f"{raw.total_signal:.4g}")
    middle.metric(
        "Signal nachher",
        f"{processed.total_signal:.4g}",
        delta=f"{100.0 * (processed.total_signal / raw.total_signal - 1.0):+.1f} %",
    )
    right.metric(
        "Rauschen (TIC, robust)",
        f"{estimate_noise_sigma(processed.tic):.4g}",
        delta=f"{estimate_noise_sigma(processed.tic) - estimate_noise_sigma(raw.tic):+.4g}",
        delta_color="inverse",
    )
    st.caption(
        "Ein stark negativer Signalunterschied ist normal: der abgezogene Untergrund "
        "macht bei einer bluten­den Säule leicht ein Vielfaches des Analytsignals aus."
    )


def _tab_analysis(cube: PyrogramDataCube, blend: dict[str, float] | None) -> None:
    """Die vollständige Auswertung: Matrix, Auflösung, Identifikation, Pass.

    Bewusst hinter einem Knopf und nicht bei jedem Rerun: eine Vollauflösung
    dauert ein bis zwei Minuten. Streamlit führt das Skript bei jeder Eingabe neu
    aus, also würde ein automatischer Start die Oberfläche unbenutzbar machen.
    """
    st.subheader("Auswertung")
    st.caption(
        "Vorverarbeitung, Abzug der Polyolefin-Matrix, Kurvenauflösung (MCR-ALS), "
        "Identifikation gegen die Marker-Bibliothek, Degradations-Indizes und der "
        "Rezyklat-Pass — in einem Durchlauf. Dauert je nach Größe des Laufs ein "
        "bis zwei Minuten."
    )

    left, right = st.columns([1, 2])
    first_carbon = left.number_input(
        "Kettenlänge des ersten Alkan-Clusters",
        min_value=0, max_value=40, value=0, step=1,
        help="0 = unbekannt. Mit Angabe wird die Kováts-Leiter darauf verankert, "
        "ohne sie über die Totzeit geschätzt — der Pass weist beides aus.",
    )
    right.checkbox(
        "Kalibriert (gravimetrische Referenzmischungen liegen vor)",
        value=False, disabled=True,
        help="In der Oberfläche bewusst gesperrt. Diese Angabe entscheidet, ob der "
        "Pass seine Prozente als quantitativ ausweist; sie gehört an den "
        "Methodennachweis, nicht an eine Checkbox.",
    )

    if st.button("Auswertung starten", type="primary"):
        st.session_state["analysis"] = _run_analysis(
            cube, int(first_carbon) or None
        )

    outcome = st.session_state.get("analysis")
    if outcome is None:
        st.info("Noch keine Auswertung gestartet.")
        return
    if isinstance(outcome, str):
        st.error(outcome)
        return

    passport, tables, html = outcome
    _render_passport(passport, tables, html, blend)


@st.cache_data(show_spinner="Auswertung läuft — Matrix, MCR-ALS, Identifikation …")
def _run_analysis(
    cube: PyrogramDataCube, first_carbon_number: int | None
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], str] | str:
    """Die Kette ausführen und in anzeigbare Strukturen überführen.

    Gibt einfache Typen zurück statt der Ergebnisobjekte: Streamlits Cache
    serialisiert, und ein ``PyrogramDataCube`` im Ergebnis würde jeden Treffer
    unnötig teuer machen. Ein Fehler kommt als Text zurück, damit er die
    Oberfläche nicht abbricht.
    """
    from pyrecycle_analytics.library.repository import MarkerLibrary
    from pyrecycle_analytics.reporting import analyse_pyrogram, render_html

    try:
        library = MarkerLibrary.in_memory()
        result = analyse_pyrogram(
            cube, library, first_carbon_number=first_carbon_number
        )
    except PyRecycleError as error:
        return f"Auswertung fehlgeschlagen: {error}"

    passport = result.passport
    tables = {
        "Polymerfraktionen": pd.DataFrame(
            [
                {
                    "Polymer": str(fraction.polymer),
                    "Anteil [%]": round(fraction.share_percent, 2),
                    "Konfidenz": str(fraction.confidence),
                    "Markermuster": fraction.marker_pattern,
                    "Marker": f"{len(fraction.markers_found)}/{fraction.markers_expected}",
                }
                for fraction in passport.polymer_fractions
            ]
            + [
                {
                    "Polymer": "nicht zugeordnet",
                    "Anteil [%]": round(passport.unassigned_share_percent, 2),
                    "Konfidenz": "—",
                    "Markermuster": "Signal, das kein erkanntes Polymer erklärt",
                    "Marker": "—",
                }
            ]
        ),
        "Identifikationen": pd.DataFrame(
            [
                {
                    "Verbindung": hit.compound_name,
                    "RT [min]": round(hit.retention_time_s / 60.0, 2),
                    "RI (Probe)": round(hit.retention_index, 0),
                    "RI (Bibliothek)": round(hit.library_retention_index, 0),
                    "Spektrum": round(hit.spectral_similarity, 3),
                    "RI-Übereinstimmung": round(hit.retention_agreement, 3),
                    "Score": round(hit.score, 3),
                    "Fläche": f"{hit.area:.4g}",
                }
                for hit in sorted(result.identifications, key=lambda h: -h.score)
            ]
        ),
        "Regulatorik": pd.DataFrame(
            [
                {
                    "Substanz": finding.substance,
                    "CAS": finding.cas_number or "—",
                    "Befund": "detektiert" if finding.detected else "nicht detektiert",
                    "Grenzwert [%]": finding.limit_percent,
                    "Bewertung": (
                        "nicht bewertbar"
                        if finding.exceeds_limit is None and finding.detected
                        else "—"
                        if finding.exceeds_limit is None
                        else "ÜBER GRENZWERT"
                        if finding.exceeds_limit
                        else "eingehalten"
                    ),
                }
                for finding in passport.regulatory_findings
            ]
        ),
    }

    degradation = result.degradation
    summary: dict[str, Any] = {
        "sample_id": passport.sample.sample_id,
        "calibration_status": str(passport.calibration_status),
        "disclaimer": passport.calibration_status.disclaimer,
        "unassessable": list(passport.unassessable_findings),
        "n_components": result.n_components,
        "n_identified": result.n_identified,
        "matrix_fraction": passport.provenance.matrix_explained_fraction,
        "ri_anchor": passport.provenance.retention_index_anchor or "keine",
        "warnings": list(result.warnings),
        "degradation": (
            {
                "Carbonyl-Index": degradation.carbonyl_index,
                "Säureanteil": degradation.acid_share,
                "Alken/Alkan": degradation.alkene_to_alkane,
                "Verzweigungsindex": degradation.branching_index,
                "mittlere Kettenlänge": degradation.mean_carbon_number,
            }
            if degradation is not None
            else None
        ),
        "degradation_notes": list(degradation.notes) if degradation is not None else [],
        "json": passport.model_dump_json(indent=2),
    }
    return summary, tables, render_html(passport)


def _render_passport(
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    html: str,
    blend: dict[str, float] | None,
) -> None:
    """Das Ergebnis anzeigen — Einschränkungen vor den Zahlen, wie im Dokument."""
    st.warning(
        f"**Kalibrierstatus: {summary['calibration_status']}** — {summary['disclaimer']}"
    )
    if summary["unassessable"]:
        st.error(
            "**Detektiert, Konformität nicht bewertbar:** "
            + ", ".join(summary["unassessable"])
            + ". Grenzwerte sind Massenanteile; ohne Kalibrierstandards wird hier "
            "keine Aussage über Einhaltung getroffen."
        )

    columns = st.columns(4)
    columns[0].metric("Komponenten aufgelöst", summary["n_components"])
    columns[1].metric("Verbindungen identifiziert", summary["n_identified"])
    columns[2].metric("Matrixanteil am Signal", f"{summary['matrix_fraction']:.1%}")
    columns[3].metric("RI-Verankerung", summary["ri_anchor"])

    st.markdown("#### Polymerfraktionen")
    st.dataframe(tables["Polymerfraktionen"], hide_index=True, width="stretch")
    st.caption(
        "Nicht zugeordnetes Signal wird ausgewiesen und **nicht** auf die erkannten "
        "Polymere verteilt — eine Verteilung würde jede Zahl proportional zu dem "
        "aufblähen, was die Methode nicht erklären konnte."
    )
    if blend:
        st.caption(
            "Bekannte Einwaage der Demo-Rezeptur (Massenanteile, nicht "
            "Signalanteile): "
            + ", ".join(f"{name} {100.0 * value:.1f} %" for name, value in blend.items())
        )

    st.markdown("#### Identifikationen")
    if tables["Identifikationen"].empty:
        st.info("Keine Verbindung erreichte die Schwelle der Bibliothek.")
    else:
        st.dataframe(tables["Identifikationen"], hide_index=True, width="stretch")

    st.markdown("#### Regulierte Substanzen")
    st.dataframe(tables["Regulatorik"], hide_index=True, width="stretch")

    st.markdown("#### Degradation")
    if summary["degradation"] is None:
        st.info("Degradations-Indizes konnten nicht berechnet werden.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Kennzahl": key, "Wert": round(value, 4)}
                    for key, value in summary["degradation"].items()
                    if value is not None
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Diese Kennzahlen sind Verhältnisse. Ohne eine Virgin-Referenz desselben "
            "Materials unter derselben Methode sind sie nicht interpretierbar und "
            "dürfen nicht über Polymere hinweg verglichen werden."
        )
        for note in summary["degradation_notes"]:
            st.caption(f"· {note}")

    if summary["warnings"]:
        with st.expander(f"Hinweise der Auswertung ({len(summary['warnings'])})"):
            for warning in summary["warnings"]:
                st.markdown(f"- {warning}")

    left, right = st.columns(2)
    left.download_button(
        "Pass als HTML",
        data=html,
        file_name=f"pass-{summary['sample_id']}.html",
        mime="text/html",
    )
    right.download_button(
        "Pass als JSON",
        data=summary["json"],
        file_name=f"pass-{summary['sample_id']}.json",
        mime="application/json",
    )


def _tab_export(cube: PyrogramDataCube) -> None:
    """Den aktuellen Stand als Archiv herunterladen."""
    st.subheader("Export")
    st.caption(
        "Das NPZ-Archiv enthält Achsen, Matrix und die vollständigen Metadaten "
        "inklusive Verarbeitungskette und Prüfsumme — self-describing und wieder "
        "über `PyrogramDataCube.load_npz()` einlesbar."
    )
    # Über save_npz statt eigener savez-Aufruf: nur so ist garantiert, dass die
    # heruntergeladene Datei von load_npz auch wieder gelesen werden kann
    # (Pydantic-computed-fields müssen vor dem Schreiben entfernt werden).
    with tempfile.TemporaryDirectory() as directory:
        path = cube.save_npz(Path(directory) / cube.metadata.sample.sample_id)
        payload = path.read_bytes()

    st.download_button(
        "Als .npz herunterladen",
        data=payload,
        file_name=f"{cube.metadata.sample.sample_id}.npz",
        mime="application/octet-stream",
    )


def main() -> None:
    """Seite aufbauen."""
    st.set_page_config(page_title="PyRecycle-Analytics", layout="wide")
    st.title("PyRecycle-Analytics")
    st.caption(
        f"Py-GC/MS-Auswertung für Post-Consumer-Rezyklate · Version {__version__} · "
        "Import, Sichtung, Vorverarbeitung, Auswertung bis zum Rezyklat-Pass"
    )

    if not pyopenms_available():
        st.sidebar.info(
            "pyopenms fehlt — .CDF funktioniert, mzML/mzXML nicht. "
            "Nachinstallieren mit `pip install 'pyrecycle-analytics[io]'`."
        )

    cube, blend = _sidebar_source()
    config = _sidebar_preprocessing()

    if cube is None:
        st.info(
            "**Eigene Messung laden:** links eine `.CDF`- (ANDI-MS/AIA netCDF) oder "
            "`.mzML`-Datei hochladen. Beides exportiert jedes gängige GC/MS — bei "
            "Agilent ChemStation/MassHunter und Shimadzu GCMSsolution heißt es "
            "*AIA*- oder *ANDI*-Export.\n\n"
            "Ohne Messdaten lässt sich die Oberfläche mit einer simulierten Probe "
            "ausprobieren (Auswahl links)."
        )
        return

    processed = preprocess(cube, config) if config is not None else None

    overview, traces, spectrum, comparison, analysis, export = st.tabs(
        [
            "Übersicht",
            "Ionenspuren",
            "Massenspektrum",
            "Vorverarbeitung",
            "Auswertung",
            "Export",
        ]
    )
    active = processed if processed is not None else cube

    with overview:
        _tab_overview(active, blend)
    with traces:
        _tab_ion_traces(active)
    with spectrum:
        _tab_spectrum(active)
    with comparison:
        _tab_preprocessing(cube, processed)
    with analysis:
        # Absichtlich der Rohlauf: analyse_pyrogram bringt seine eigene
        # Vorverarbeitung mit und protokolliert sie im Pass. Ein bereits
        # vorverarbeiteter Cube würde zweimal baseline-korrigiert und die
        # Provenienz im Pass stimmte nicht mehr mit dem überein, was gerechnet
        # wurde.
        _tab_analysis(cube, blend)
    with export:
        _tab_export(active)


if __name__ == "__main__":
    main()
