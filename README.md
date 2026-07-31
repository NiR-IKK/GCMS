# PyRecycle-Analytics

Py-GC/MS-Auswerteplattform für die **inverse Polymeranalytik** von Post-Consumer-Rezyklaten (PCR).

Ziel ist nicht generische Chromatographie-Software, sondern der Rückschluss von überlappenden
Pyrolysaten auf polymere Hauptkomponenten, Additive und Degradationsgrade.

> **Status: Meilensteine 1 bis 4 abgeschlossen.** Von der Rohdatei bis zum Rezyklat-Pass läuft die
> Kette durch: Ingestion, Preprocessing, Matrix-Subtraktion, MCR-ALS, Marker-Bibliothek,
> Identifikation, Degradations-Index, Pass als JSON/HTML/PDF und eine FastAPI-Schnittstelle.
> 865 Tests laufen grün. Was die Kette auf dem eigenen Benchmark tatsächlich leistet — und wo sie
> ausdrücklich an ihre Grenzen kommt — steht gemessen in [Abschnitt 6](#6-was-die-kette-leistet-und-wo-sie-aufhört).
> Offen bleibt bewusst nur MS2.4 (PARAFAC2-Mehrlaufauswertung), von Anfang an als optional geplant.

---

## 0. Eigene Messungen auswerten

Die simulierten Proben sind **Prüfmittel, nicht das Produkt**. Der Ingestion-Layer ist für echte
Messdateien gebaut; der Generator liegt bewusst in `tests/`, weil ohne ihn keine Deconvolution
automatisiert prüfbar wäre.

```bash
pip install -e '.[all]'
streamlit run streamlit_app.py     # links "Eigene Messung hochladen"
```

Oder direkt aus Python, von der Rohdatei bis zum fertigen Pass:

```python
from pyrecycle_analytics.ingestion import read_pyrogram
from pyrecycle_analytics.library.repository import MarkerLibrary
from pyrecycle_analytics.reporting import analyse_pyrogram, render_pdf

cube = read_pyrogram("Probe_4711.CDF")
result = analyse_pyrogram(cube, MarkerLibrary.in_memory())

for fraction in result.passport.polymer_fractions:
    print(fraction.polymer, f"{fraction.share_percent:.1f} %", fraction.confidence)
print("nicht zugeordnet:", f"{result.passport.unassigned_share_percent:.1f} %")

render_pdf(result.passport, "Probe_4711_Pass.pdf")
```

Oder als Dienst:

```bash
uvicorn pyrecycle_analytics.api.app:create_app --factory     # OpenAPI unter /docs
```

**Welches Exportformat?** `.CDF` (ANDI-MS / AIA netCDF) exportiert jedes gängige GC/MS — bei
Agilent ChemStation und MassHunter sowie bei Shimadzu GCMSsolution heißt der Menüpunkt *AIA*- oder
*ANDI*-Export. Alternativ `.mzML` oder `.mzXML`, etwa über msConvert. Achtung: ein
netCDF-**4**/HDF5-Container mit `.cdf`-Endung ist kein ANDI-MS; die Software sagt das mit
Lösungshinweis, statt kryptisch abzustürzen.

**Was der Pass ausdrücklich nicht ist.** Ohne gravimetrisch angesetzte Referenzmischungen, die
unter derselben Methode gemessen wurden, sind alle Prozentangaben semiquantitativ. Das Dokument
schreibt das als Kalibrierstatus über die Zahlen, nicht in eine Fußnote darunter — und ein
regulatorischer Nachweis („DEHP detektiert") wird strikt von einer Konformitätsaussage getrennt,
die er ohne Kalibrierstandards nicht trägt.

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
├── truth.py                     ComponentTruth, DriftTruth, PyrogramTruth (Benchmark-Referenz)
└── passport.py                  RecyclatePassport, Kalibrierstatus, Regulatorik-Befunde

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
├── validation/                  Bewertungs-Harness: ungarische Zuordnung, Kennzahlen,
│                                triviale Vergleichsbasis
├── matrix/
│   └── polyolefin.py            Homologenreihen-Modell und subtract_polymer_matrix
├── deconvolution/
│   ├── windows.py               Fensterbildung auf TIC + kanalweiser Aktivität
│   ├── rank.py                  Parallelanalyse, EFA, Malinowski-IND, Konsens + Warnung
│   ├── mcrals.py                SIMPLISMA-Start, ALS mit Nichtnegativität/Unimodalität
│   └── pipeline.py              resolve_pyrogram: Matrix-Abzug → Fenster → Rang → MCR-ALS
├── library/
│   ├── schema.py                SQLAlchemy-2.0-Modell (Compound, Spektrum, RI, Muster)
│   ├── reference_data.py        Kuratierte Bibliothek — bewusst getrennt von tests/
│   ├── repository.py            MarkerLibrary: lesende Sicht ohne offene Session
│   └── retention_index.py       Kováts-Leiter aus dem Alkan-Kamm der Probe selbst
├── identification/engine.py     Spektrenähnlichkeit + RI + Marker-Muster → Konfidenz
├── degradation/engine.py        Carbonyl-Index, Säureanteil, Alken/Alkan, Verzweigung
├── reporting/
│   ├── analysis.py              analyse_pyrogram: die vollständige Kette
│   ├── passport.py              build_passport, MatrixPolyolefinEvidence
│   ├── reach.py                 Wachliste + Notiz zum nicht GC-gängigen Teil
│   └── render.py                Deterministisches HTML, PDF via WeasyPrint
├── api/app.py                   FastAPI: Upload, Analyse-Job, Pass, Bibliothek
└── exceptions.py

streamlit_app.py                 Oberfläche: Upload, Sichtung, Vorverarbeitung, Export

tests/
├── synthetic_data.py            SyntheticPyrogramGenerator (TDD-Basis)
├── reference_spectra.py         EI-Spektren und Retentionsanker der Marker
└── test_*.py                    865 Tests
```

---

## 3. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'         # oder gezielt: io, db, api, report, ui, dev
pytest                          # 865 Tests, ca. 7 min
streamlit run streamlit_app.py
```

| Extra | Wofür | Ohne das Extra |
|---|---|---|
| `io` | pyopenms — mzML / mzXML / mzData | `.CDF` liest scipy; die mzML-Tests überspringen sich selbst |
| `db` | SQLAlchemy — Marker-Bibliothek | keine Identifikation, kein Pass |
| `api` | FastAPI, uvicorn, python-multipart | `analyse_pyrogram(...)` läuft weiter ohne Server |
| `report` | WeasyPrint — PDF | HTML und JSON bleiben nutzbar |
| `ui` | Streamlit, Plotly | die Bibliotheks-API bleibt unberührt |

`pyopenms` und WeasyPrint werden erst beim Aufruf importiert und fehlen laut: die betroffene
Funktion wirft eine `MissingDependencyError`, die Paketname, Zweck und den passenden
Installationsbefehl nennt, statt eines nackten `ImportError` tief in der Kette. Die übrigen Extras
sind Importvoraussetzung ihres Teilpakets — ohne `db` lässt sich `library` nicht importieren.

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

Das Retentionsfenster akzeptiert beide Schreibweisen — positional oder als Tupel, analog zu
`mz_range`; ohne Angabe bleiben alle Scans erhalten:

```python
cube.window(640.0, 700.0, mz_range=(50.0, 300.0))
cube.window(rt_range_s=(640.0, 700.0), mz_range=(50.0, 300.0))
cube.window(mz_range=(50.0, 300.0))
```

Uploads liegen als Bytes vor, nicht als Pfad — beide Parser brauchen aber eine echte Datei:

```python
from pyrecycle_analytics.ingestion import read_pyrogram_bytes

cube = read_pyrogram_bytes(upload.getvalue(), upload.name)
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

### Matrix-Subtraktion und Deconvolution

```python
from pyrecycle_analytics.matrix.polyolefin import subtract_polymer_matrix
from pyrecycle_analytics.deconvolution import DeconvolutionConfig, resolve_pyrogram

matrix = subtract_polymer_matrix(cube)
matrix.explained_fraction        # Signalanteil, den die Homologenreihe erklärt
matrix.residual                  # PyrogramDataCube ohne die Matrix
matrix.model.detection.apex_times_s   # Kamm-Positionen — die Kováts-Leiter für MS3
matrix.diagnostics["n_clusters"]      # erkannte Kettenlängen, kontaminierte Positionen …

report = resolve_pyrogram(cube, DeconvolutionConfig())
report.result.S                  # Reinspektren, L1-normiert
report.result.areas              # Fläche je aufgelöster Komponente
report.result.apex_times         # Retentionszeit je Komponente
report.result.lack_of_fit        # LOF in %
```

`resolve_pyrogram` zieht die Matrix zuerst ab und legt MCR-ALS auf das Residuum — nicht umgekehrt.
Die Begründung steht gemessen in [ROADMAP.md](ROADMAP.md), Befund 2.

### Identifikation, Degradation, Pass

```python
from pyrecycle_analytics.library.repository import MarkerLibrary
from pyrecycle_analytics.library.retention_index import calibrate_from_comb
from pyrecycle_analytics.identification import identify_compounds, identify_polymers
from pyrecycle_analytics.degradation import compute_degradation_indices

library = MarkerLibrary.in_memory()      # oder MarkerLibrary(create_library("postgresql://…"))

# Der Kamm liefert die Abstände, nicht die absoluten Kettenlängen. Ist die
# Kohlenstoffzahl des ersten Clusters bekannt, wird die Leiter darauf verankert;
# sonst wird sie über die Totzeit geschätzt und der Pass weist das aus.
ladder = calibrate_from_comb(matrix.model.detection.apex_times_s, first_carbon_number=8)

hits = identify_compounds(report.result, library, calibration=ladder)
polymers = identify_polymers(hits, library)
indices = compute_degradation_indices(cube, matrix_model=matrix.model)
indices.branching_index          # iso-Alkene / n-Alkane — trennt PP, LDPE und HDPE
```

Die Bibliothek liegt in `library/reference_data.py` und ist **nicht** aus
`tests/reference_spectra.py` abgeleitet. Ein Test prüft aktiv, dass beide Tabellen auseinandergehen
— sonst validierte sich die Identifikation gegen ihre eigene Quelle.

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

**Die Matrix wird vor der Kurvenauflösung abgezogen, nicht danach.** Der ursprüngliche Auftrag
nennt die umgekehrte Reihenfolge. Gemessen sinkt die Komponentenzahl je koeluierendem Fenster durch
den Abzug von median 7 (max 12) auf median 3, und rund ein Drittel der Fenster wird ganz leer.
MCR-ALS mit 7–12 nahezu kollinearen Komponenten ist nicht durch Nichtnegativität und Unimodalität
einzugrenzen; mit 3 ist es Routine. Begründung und Zahlen: [ROADMAP.md](ROADMAP.md), Befund 2.

**Der Säure-Kanal ist m/z 60, nicht m/z 73.** m/z 73 ist zwar ein Fragment kurzkettiger Säuren,
aber auch das dominante Siloxan-Ion des Säulenblutens. Ein darauf gestützter Säureanteil misst im
Wesentlichen die Säule. Das kostet Empfindlichkeit und ist die einzige Wahl, die eine Kennzahl
liefert, die tatsächlich die Probe beschreibt.

**Die Kontaminationsreparatur vergleicht entlang der Kohlenstoffachse, nicht über den Cosinus.**
Ein Clusterspektrum, das einen koeluierenden Analyten enthält, hebt diesen beim Abzug fast exakt
auf — ohne Reparatur bleiben von ε-Caprolactam 2 % übrig, mit 88 %. Eine cosinusbasierte Prüfung
löste das nicht, weil das kontaminierte Spektrum den Nachbarn immer noch ähnlich sieht. Verglichen
wird deshalb kanalweise gegen dieselbe Triplett-Position der Nachbarketten.

**Nicht zugeordnetes Signal wird ausgewiesen, nicht verteilt.** Und der Kalibrierstatus eines
Passes ist immer der schwächste seiner Fraktionen: eine gut belegte Fraktion darf eine schlecht
belegte nicht aufwerten. Beides ist im Schema erzwungen, nicht nur in der Darstellung.

---

## 6. Was die Kette leistet, und wo sie aufhört

Alle Zahlen hier sind auf dem eigenen Benchmark gemessen (Seed 11, Vollauflösung), nicht geschätzt.
Die Messskripte sind der Bewertungs-Harness aus `validation/`; die Tests halten die Aussagen fest,
damit sie bei einer Änderung nicht stillschweigend verfallen.

### Deconvolution gegen die triviale Vergleichsbasis

Ohne diesen Vergleich wäre nicht belegbar, dass MCR-ALS überhaupt etwas verbessert. Die Basis ist
Peakerkennung auf dem TIC mit Spektrum am Apex — was ein Anwender ohne Chemometrie täte.

| Rezeptur | Verfahren | Recall | Cosinus (med) | Flächenfehler (med) | LOF % | R² |
|---|---|---|---|---|---|---|
| `coelution_stress` | naiv | 0,16 | 0,983 | 6,93 | 128,6 | −0,88 |
| `coelution_stress` | **MCR-ALS** | **0,34** | 0,942 | **0,82** | **87,6** | **0,13** |
| `pcr_mixed_polyolefin` | naiv | 0,16 | 0,948 | 5,05 | 134,0 | −1,00 |
| `pcr_mixed_polyolefin` | **MCR-ALS** | **0,40** | 0,947 | **0,75** | **80,5** | **0,28** |
| `trace_pet_in_polyolefin` | naiv | 0,15 | 0,989 | 4,07 | 122,7 | −0,67 |
| `trace_pet_in_polyolefin` | **MCR-ALS** | **0,35** | 0,924 | **0,93** | **94,3** | **0,01** |

Der hohe Cosinus der naiven Basis ist kein Erfolg: sie findet fast nur die großen, gut getrennten
Alkan-Peaks, deren Spektrum am Apex ohnehin sauber ist. Der Flächenfehler von 4 bis 7 (also 400 bis
700 %) zeigt, was in den koeluierenden Fenstern passiert. Der Recall bleibt auch mit MCR-ALS bei
0,34–0,40, und das ist erwartet: die Homologenreihe wird bewusst als strukturiertes Ganzes
abgezogen statt als 120 Einzelkomponenten aufgelöst.

### Die Zahl, auf die es ankommt: benannte Marker

| Rezeptur | naiv | MCR-ALS |
|---|---|---|
| `coelution_stress` | 7 / 12 | **9 / 12** |
| `pcr_mixed_polyolefin` | 6 / 24 | **12 / 24** |
| `trace_pet_in_polyolefin` | 0 / 6 | **1 / 6** |

Auf `pcr_mixed_polyolefin` findet MCR-ALS unter anderem ε-Caprolactam (0,95 % Signalanteil,
Cosinus 0,89), Benzoesäure (0,69 %, 0,99), Divinylterephthalat (0,19 %, 0,99) und DEHP (0,15 %,
0,99) — alle vier findet die naive Basis nicht, weil sie unter einem Alkan-Cluster liegen.

**Die Nachweisgrenze liegt gemessen bei etwa 0,05 % Signalanteil.** Darunter wird nichts mehr
gefunden: Naphthalin (0,02 %), das Benzotriazol-Fragment (0,02 %), Chlorbenzol (0,01 %) und BHT
(0,04 %) bleiben in `pcr_mixed_polyolefin` unerkannt.
Auf `trace_pet_in_polyolefin` — PET bei 0,2 % Massenanteil, seine
Marker damit bei 0,03 bis 0,11 % des Signals — wird **kein einziger PET-Marker** zurückgewonnen.
Das verfehlt das in der Roadmap gesetzte Ziel („PET-Markerfläche ±25 %") deutlich, und die Zahl
steht hier, statt das Ziel nachträglich abzusenken.

### Rezyklat-Pass gegen die bekannte Rezeptur

| Rezeptur | Wahrheit (Flächenanteil) | Pass | nicht zugeordnet |
|---|---|---|---|
| `virgin_pp` | PP 100 % | PP 75,3 % (high) | 24,7 % |
| `pcr_ps_with_traces` | PS 92,8 %, PE-HD 4,6 %, PP 2,2 % | PS 79,9 % (high), PE-LD 7,1 % (medium) | 13,0 % |
| `trace_pet_in_polyolefin` | PE-LD 95,0 %, PP 4,8 %, PET 0,2 % | PE-LD 74,1 % (high), PP 0,1 % (indicative) | 25,8 % |
| `pcr_mixed_polyolefin` | PE-LD 59,6 %, PP 31,2 %, PS 6,2 %, PET 1,3 %, PA6 1,0 % | PE-LD 45,5 % (high), PP 5,5 % (indicative), PS 4,1 % (medium) | 44,9 % |

Drei Dinge sind daran abzulesen:

1. **Die Rangfolge stimmt, die absolute Höhe nicht.** Der Pass unterschätzt jede Fraktion, weil
   nicht zugeordnetes Signal ausgewiesen und **nicht** auf die erkannten Polymere verteilt wird.
   Eine Verteilung würde jede Zahl proportional zu dem aufblähen, was die Methode nicht erklären
   konnte — das wäre die unehrlichere Darstellung.
2. **PE und PP in derselben Probe sind die eigentliche Schwäche.** Beide liefern denselben
   Alkan/Alken-Kamm; die Trennung läuft allein über den Verzweigungsindex, der einen Mischkamm zu
   einem einzigen Polymer erklärt. Auf `pcr_mixed_polyolefin` (PP 31 % wahr) landet PP bei 5,5 %
   und ist als `indicative` gekennzeichnet. In der reinen Probe (`virgin_pp`) ist die Zuordnung
   korrekt. Eine echte Ko-Quantifizierung im Kamm ist offen.
3. **Die Sortenangabe bei PE ist ein Hinweis, kein Befund.** Auf `pcr_ps_with_traces` meldet der
   Pass PE-LD, wahr ist PE-HD — der Anteil (7,1 % gegen 6,8 % wahr) stimmt, die Sorte nicht. Der
   Pass markiert das entsprechend.

### Die Verankerung der Retentionsskala

Die Kováts-Leiter wird aus dem Alkan-Kamm der Probe selbst gebaut, nicht aus einem separaten
Standard. Auf dem synthetischen Benchmark weicht sie im Median um rund 140 RI-Einheiten von den
tabellierten Werten ab, im Extremfall um über 1000. Das ist **kein Fehler der Verankerung, sondern
ein Konstruktionskonflikt des Generators**: er setzt Marker gezielt auf Alkan-Cluster, um Koelution
zu erzeugen — genau dadurch kann ihre Retentionszeit nicht gleichzeitig Kováts-konsistent sein. Ein
Test hält die Abweichung fest, damit sie nicht stillschweigend verschwindet und nicht mit einer
Schwäche der RI-Logik verwechselt wird. Auf echten Messungen gilt die Einschränkung nicht.

---

## 7. Tests

```bash
pytest                      # alle Tests
pytest -m "not slow"        # ohne die Tests in Produktionsauflösung
pytest --cov=src/pyrecycle_analytics --cov=data_schemas
ruff check . && mypy        # Lint und Typprüfung: sauber
```

865 Tests. Sie prüfen drei Ebenen:

* **Mathematischer Vertrag** — exakte Bilinearität, Flächen­normierung der EMG über den gesamten
  Tailing-Bereich, L1-normierte Spektren, prozessübergreifende Reproduzierbarkeit,
  TIC-Erhaltung durch Binning und Datei-I/O.
* **Chemische Realität** — PE liefert verschmolzene Tripletts, LDPE ist mehr als dreimal so stark
  verzweigt wie HDPE, PP wird vom C9-Trimer dominiert, Alterung hebt Carbonyl-Gehalt und
  iso-Alken-Anteil, die Styrol-Triade liegt bei 100 : 12 : 6, und der PET-Spurenmarker ist im TIC
  unsichtbar, auf m/z 105 aber klar detektierbar.
* **Ehrlichkeit der Aussage** — ein unkalibrierter Lauf darf nirgends als kalibriert erscheinen;
  ohne Kalibrierung wird kein Unsicherheitsintervall ausgewiesen; nicht zugeordnetes Signal wird
  ausgewiesen statt verteilt und die Summe der Anteile kann 100 % nicht überschreiten;
  „detektiert" wird nie zu „konform"; und die Bibliothek unter `library/` ist nachweislich nicht
  identisch mit `tests/reference_spectra.py`.

Die letzten beiden Punkte sind Absicht. Wenn ein Benchmark einfacher wäre als die Realität, würde
MCR-ALS ihn aus den falschen Gründen bestehen — mehrere Tests sichern deshalb explizit ab, dass die
*Schwierigkeit* vorhanden ist. Und da das Ergebnis ein Dokument mit dem Wort „Pass" im Namen ist,
sind die Einschränkungen selbst testpflichtig, nicht nur die Zahlen.

---

## 8. Stand und was offen bleibt

Die Kette MS1 → MS4 ist vollständig; die ausführliche Planung mit den gemessenen Ausgangszahlen,
Akzeptanzkriterien und Risiken steht in **[ROADMAP.md](ROADMAP.md)**.

**Umgesetzt.** Ingestion (ANDI-MS, mzML/mzXML/mzData) mit Provenienz · kanalweises Preprocessing ·
Bewertungs-Harness mit trivialer Vergleichsbasis · Fensterbildung und Rangschätzung mit
Divergenzwarnung · Polyolefin-Matrixmodell und `subtract_polymer_matrix` · MCR-ALS mit
SIMPLISMA-Start, Nichtnegativität und Unimodalität · Marker-Bibliothek (SQLAlchemy 2.0) ·
Kováts-Verankerung aus dem Kamm der Probe · Identifikation mit Konfidenz · Degradations-Indizes ·
Rezyklat-Pass als JSON/HTML/PDF · FastAPI-Dienst.

**Bewusst nicht umgesetzt.** MS2.4 (PARAFAC2-Mehrlaufauswertung, `generate_series()`) war von
Anfang an als optional geplant und ist für einen Einzellauf-Pass nicht erforderlich. Die
Voraussetzung dafür steht im Generator bereit.

**Offene Punkte, nach Wirkung sortiert.**

1. **PE/PP-Ko-Quantifizierung im gemeinsamen Kamm** — der größte Einzelfehler des Passes
   (siehe Abschnitt 6). Ein Ansatz wäre, den Kamm in zwei Reihen mit eigenen Amplituden zu
   zerlegen, statt ihn über einen skalaren Verzweigungsindex einem Polymer zuzuschlagen.
2. **Nachweisgrenze unter 0,05 % Signalanteil** — betrifft PET in stark polyolefindominierten
   Rezyklaten und die meisten Additivmarker. Zielionen-gestützte Fenster statt TIC-gestützter
   wären der nächste Hebel.
3. **Persistenz der Analysen** — Ergebnisse liegen im API-Prozess, ein Neustart verliert sie. Das
   Bibliotheksschema hat bereits die Tabellen dafür.
4. **Streamlit-Oberfläche** — zeigt bislang Import, Sichtung und Vorverarbeitung; aufgelöste
   Profile, Identifikationstabelle und Pass-Vorschau fehlen noch.
5. **Echte Kalibrierung** — solange keine gravimetrischen Referenzmischungen unter derselben
   Methode gemessen wurden, bleibt jeder Pass `response-corrected` und sagt das auch.
