# PyRecycle-Analytics

Py-GC/MS-Auswerteplattform für die **inverse Polymeranalytik** von Post-Consumer-Rezyklaten (PCR).

Ziel ist nicht generische Chromatographie-Software, sondern der Rückschluss von überlappenden
Pyrolysaten auf polymere Hauptkomponenten, Additive und Degradationsgrade.

> **Status: Meilenstein 1 abgeschlossen.** Ingestion-Layer, Datenmodell, Preprocessing und der
> synthetische Pyrogramm-Generator stehen; 612 Tests laufen grün. Meilenstein 2 (MCR-ALS &
> Matrix-Subtraktion) setzt direkt auf dem hier erzeugten Ground Truth auf.

---

## 1. Warum ein synthetischer Generator zuerst?

Von keiner realen PCR-Probe ist die wahre Zusammensetzung bekannt. Damit gibt es keine Referenz,
gegen die ein Deconvolution-Ergebnis gemessen werden könnte. `tests/synthetic_data.py` liefert
genau diese fehlende Referenz: Aus einer *deklarierten* Polymer-Rezeptur wird ein Pyrogramm
gebaut und zusammen mit der exakten bilinearen Faktorisierung zurückgegeben, die es erzeugt hat.

```python
from tests.synthetic_data import SyntheticPyrogramGenerator, RECIPES

sample = SyntheticPyrogramGenerator(seed=42).generate(RECIPES["pcr_mixed_polyolefin"])

sample.cube          # PyrogramDataCube — was die Pipeline sieht (mit Baseline + Rauschen)
sample.truth         # PyrogramTruth  — jede Komponente, Fläche, RT, Polymer-Zuordnung
sample.C, sample.S   # Elutionsprofile und Reinspektren — das Zielergebnis der Deconvolution
sample.clean         # das rausch- und baselinefreie Signal C @ S
```

Für den rauschfreien Fall gilt **`D == C @ S` bis auf Maschinengenauigkeit** — das Modell, das
MCR-ALS und PARAFAC2 annehmen. Ground Truth ist damit keine Näherung, sondern exakt die
Faktorisierung, die der Algorithmus rekonstruieren soll.

### Was bewusst simuliert wird — und warum

| Effekt | Warum er für die Bewertung nötig ist |
|---|---|
| **Homologe Reihen** (Alkan/Alken/Dien-Tripletts C6–C34) | Benachbarte Kettenlängen haben nahezu identische Spektren (Cosinus-Ähnlichkeit > 0.98). Diese Rangdefizienz ist die eigentliche numerische Schwierigkeit einer Polyolefin-Matrix. |
| **Gezielte Koelution** | Styrol auf dem C8-Cluster, Caprolactam auf C14, Benzoesäure auf C13, PP-Trimer-Marker zwischen C8 und C9 — R < 1, also real verschmolzen. |
| **Retentionsdrift** | Glatte, nichtlineare Zeitachsen-Verzerrung (Offset + Stretch + Krümmung + Oszillation). Späte Peaks driften stärker als frühe — genau die Bedingung, für die PARAFAC2 existiert. |
| **Säulenbluten** | Der ansteigende Untergrund liegt auf den Siloxan-Ionen (m/z 73, 147, 207, 221, 281), nicht flach verteilt. m/z 147 stört direkt den Phthalat-/PET-Quantifier bei m/z 149. |
| **Degradation** | Alterungsgrad je Polymer verschiebt iso-Alken/n-Alkan-Verhältnis, erzeugt Ketone/Aldehyde/Säuren und verzerrt die Oligomer-Triade. |
| **Response-Faktoren** | Massenanteil ≠ Signalanteil. PVC verliert den Großteil als HCl, PET/PA bilden polare, teils nicht eluierende Fragmente. 1 % PVC (m/m) ergibt ≈ 0.3 % des Chromatogramms. |

### Verfügbare Rezepturen

`virgin_hdpe`, `virgin_ldpe`, `virgin_pp`, `virgin_ps`, `aged_hdpe`, `pcr_mixed_polyolefin`,
`pcr_ps_with_traces`, `weee_flame_retarded`, `coelution_stress`, `trace_pet_in_polyolefin`.

Mehrere Läufe derselben Mischung mit unterschiedlicher Drift (Input für PARAFAC2/Alignment):

```python
runs = generator.generate_series(RECIPES["pcr_mixed_polyolefin"], n_runs=5)
```

---

## 2. Projektstruktur

```
data_schemas/                    Pydantic-Datenmodell (ohne schwere Abhängigkeiten)
├── enums.py                     PolymerClass, MarkerRole, RecyclateStream, SourceFormat
├── acquisition.py               Pyrolyse-, GC- und MS-Bedingungen, Probenmetadaten
├── pyrogram.py                  PyrogramMetadata, MzAxisSpec, Preprocessing-Audit-Trail
└── truth.py                     ComponentTruth, DriftTruth, PyrogramTruth (Benchmark-Referenz)

src/pyrecycle_analytics/
├── core/
│   ├── datacube.py              PyrogramDataCube: RT × m/z × Intensität + Provenienz
│   ├── peakshapes.py            EMG/Gauss, Profilmatrix, chromatographische Auflösung
│   └── binning.py               Ragged Scans → rechteckige Matrix (nominale Massenachse)
├── ingestion/
│   ├── registry.py              Formaterkennung (Magic Bytes → XML-Root → Endung)
│   ├── mzml.py                  mzML / mzXML / mzData via pyopenms
│   ├── andi_cdf.py              ANDI-MS / AIA netCDF via scipy.io.netcdf_file
│   ├── writers.py               Export nach mzML und ANDI-CDF
│   └── common.py                Gemeinsamer Zusammenbau + Reader-Warnungen
├── preprocessing/
│   ├── baseline.py              AsLS (Eilers/Boelens) und SNIP, kanalweise
│   ├── smoothing.py             Savitzky-Golay, Gauss, robuste Rauschschätzung
│   └── pipeline.py              Deklarative, reproduzierbare Preprocessing-Kette
└── exceptions.py

tests/
├── synthetic_data.py            SyntheticPyrogramGenerator (TDD-Basis)
├── reference_spectra.py         EI-Spektren und Retentionsanker der Marker
└── test_*.py                    612 Tests
```

---

## 3. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[io,dev]'      # io = pyopenms für die XML-Formate
pytest                          # 612 Tests, ca. 25 s
```

Ohne `pyopenms` bleibt alles außer dem mzML/mzXML/mzData-Reader nutzbar; die entsprechenden
Tests überspringen sich selbst, und der Reader wirft eine `MissingDependencyError` mit dem
passenden Installationsbefehl.

---

## 4. Nutzung

### Rohdaten einlesen

```python
from pyrecycle_analytics.ingestion import read_pyrogram

cube = read_pyrogram("PCR_LDPE_batch7.CDF", mz_range=(35.0, 550.0), rt_range_s=(180.0, 3600.0))

cube.shape                  # (n_scans, n_mz)
cube.tic                    # Totalionenstrom
cube.eic(105.0)             # Extracted-Ion-Chromatogramm (PET-Marker Benzoesäure)
cube.eic_sum([104, 103, 78])# summierte Diagnoseionen (Styrol)
cube.window(640.0, 700.0)   # lokales Fenster für die Kurvenauflösung
cube.metadata.reader_warnings
```

Das Format wird über Magic Bytes erkannt, nicht über die Endung: eine als `.mzML` benannte
netCDF-Datei wird korrekt als ANDI-MS gelesen, und ein netCDF-4/HDF5-Container mit `.cdf`-Endung
liefert eine Fehlermeldung, die den Ausweg nennt, statt eines Parser-Absturzes.

### Preprocessing

```python
from pyrecycle_analytics.preprocessing import PreprocessingConfig, preprocess

config = PreprocessingConfig(
    rt_range_s=(180.0, 3400.0),
    smoothing="savgol", savgol_window=7,
    baseline="asls", asls_lam=1e7, asls_p=0.005,
)
processed = preprocess(cube, config)
[step.name for step in processed.metadata.preprocessing]
# ['window', 'smooth:savgol', 'baseline:asls']
```

Die Kette ist als validiertes Pydantic-Objekt beschrieben und wird Schritt für Schritt im
Audit-Trail des Cubes protokolliert — ein Ergebnis lässt sich Monate später exakt reproduzieren.

---

## 5. Bewusste Entwurfsentscheidungen

**ANDI-MS wird nicht über pyopenms gelesen.** pyopenms deckt die Proteomik-XML-Familie ab
(mzML, mzXML, mzData) und hat keinen netCDF-Loader. ANDI-MS (ASTM E1947) ist netCDF-3 classic,
das `scipy.io.netcdf_file` direkt liest — ohne Zusatzabhängigkeit und ohne Hersteller-SDK. Da
`.CDF` das Format ist, das jedes GC/MS in einem Polymerlabor exportieren kann, wäre ein Verzicht
darauf keine Option gewesen.

**Baseline-Korrektur erfolgt kanalweise, nicht auf dem TIC.** Eine über alle Kanäle skalierte
TIC-Baseline verzerrt Massenspektren. Jeder Kanal hat eigene Untergrundchemie — das Bluten liegt
auf den Siloxan-Ionen, der unaufgelöste Polyolefin-Hügel auf den Alkyl-Ionen.

**Geglättet wird nur entlang der Zeitachse.** Benachbarte Nominalmassen sind chemisch unabhängig
(m/z 56 und 57 stammen aus verschiedenen Fragmentierungswegen); eine Glättung über die m/z-Achse
zerstört genau die Feinstruktur, auf der die Marker-Logik beruht.

**Gleiche Verbindungen aus verschiedenen Polymeren werden zusammengeführt.** n-Nonan aus PE und
aus PP ist ein Molekül und ergibt einen Peak. Zwei Truth-Einträge mit identischem Spektrum und
identischer RT wären nicht unterscheidbar — jede Recovery-Metrik würde gegen ein nicht
identifizierbares Ziel messen. Die Aufteilung bleibt in `ComponentTruth.source_areas` erhalten.

**Peaks, die durch Drift aus dem Messfenster wandern, verschwinden aus der Truth.** Sie wurden
nie gemessen; sie in der Referenz zu behalten würde jede Metrik für etwas bestrafen, das kein
Algorithmus finden kann.

**Reproduzierbarkeit über Prozessgrenzen.** Rezepturnamen gehen per `zlib.crc32` in den Seed ein,
nicht per `hash()` — Pythons String-Hashing ist prozessweise gesalzen, ein damit erzeugter
Benchmark wäre nicht regenerierbar.

---

## 6. Tests

```bash
pytest                      # alle Tests
pytest -m "not slow"        # ohne die Tests in Produktionsauflösung
pytest --cov=src/pyrecycle_analytics --cov=data_schemas
ruff check . && mypy        # Lint und Typprüfung: sauber
```

612 Tests, 96 % Abdeckung. Die Tests prüfen zwei Ebenen:

* **Mathematischer Vertrag** — exakte Bilinearität, Flächen­normierung der EMG über den gesamten
  Tailing-Bereich, L1-normierte Spektren, prozessübergreifende Reproduzierbarkeit,
  TIC-Erhaltung durch Binning und Datei-I/O.
* **Chemische Realität** — PE liefert verschmolzene Tripletts, LDPE ist mehr als dreimal so stark
  verzweigt wie HDPE, PP wird vom C9-Trimer dominiert, Alterung hebt Carbonyl-Gehalt und
  iso-Alken-Anteil, die Styrol-Triade liegt bei 100 : 12 : 6, und der PET-Spurenmarker ist im TIC
  unsichtbar, auf m/z 105 aber klar detektierbar.

Der letzte Punkt ist Absicht: Wenn ein Benchmark einfacher wäre als die Realität, würde MCR-ALS
ihn aus den falschen Gründen bestehen. Mehrere Tests sichern deshalb explizit ab, dass die
*Schwierigkeit* vorhanden ist.

---

## 7. Nächste Meilensteine

**MS2 — Deconvolution-Engine.** MCR-ALS über Fenster koeluierender Cluster (Initialisierung via
SIMPLISMA/EFA, Nichtnegativität, Unimodalität), plus
`subtract_polymer_matrix(raw_data, matrix_type="PE_PP_Backbone")`: Abzug der Alkan/Alken-
Homologenreihen zum Freilegen der Spurenfragmente. Bewertung gegen `sample.C` / `sample.S` und
gegen `truth.coeluting_groups()`; die Rezeptur `trace_pet_in_polyolefin` ist der Zielfall.

**MS3 — Marker-Bibliothek & Degradations-Index.** Relationales Schema (SQLAlchemy) für
quantitative Marker-Triaden; `DegradationEngine` auf Basis von iso-Alken/n-Alkan-Verhältnis und
Carbonyl-Markern. Die Referenzspektren unter `tests/` bleiben bewusst getrennt von der
Identifikationsbibliothek, damit diese nicht gegen ihre eigene Quelle validiert wird.

**MS4 — Digitaler Rezyklat-Pass.** FastAPI-Endpunkt mit Hauptfraktionen in %, REACH-Markern sowie
Degradations- und Verzweigungsindex. Die Response-Faktor-Korrektur aus MS1 ist Voraussetzung für
belastbare Prozentangaben.
