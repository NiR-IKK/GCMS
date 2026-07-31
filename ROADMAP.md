# Roadmap Meilenstein 2 – 4

Stand: Meilenstein 1 abgeschlossen (632 Tests grün). Dieses Dokument plant die verbleibenden
Meilensteine. Alle Zahlen darin sind **auf dem eigenen Benchmark gemessen**, nicht geschätzt — das
Messskript steht in Abschnitt 0.

---

## 0. Was die Messung ergeben hat, und warum sie den Plan ändert

Vor der Planung wurde die tatsächliche Schwierigkeit auf den Rezepturen `coelution_stress`,
`pcr_mixed_polyolefin` und `trace_pet_in_polyolefin` vermessen (Vollauflösung, 5 Hz, 8701 Scans,
372 Kanäle, 179–191 Komponenten).

### Befund 1 — Die Fenster sind zu voll für MCR-ALS

| Rezeptur | Komponenten | Cluster | Cluster­größe median / p90 / max |
|---|---|---|---|
| `coelution_stress` | 179 | 31 | 7 / 8 / **11** |
| `pcr_mixed_polyolefin` | 191 | 31 | 7 / 9 / **12** |
| `trace_pet_in_polyolefin` | 173 | 29 | 7 / 8 / 9 |

Publizierte MCR-ALS-Anwendungen arbeiten typischerweise mit 2–6 Komponenten pro Fenster. Bei 7–12
nahezu kollinearen Komponenten ist die Rotationsmehrdeutigkeit nicht mehr durch Nichtnegativität
und Unimodalität einzugrenzen.

### Befund 2 — Matrix-Subtraktion *vor* MCR-ALS löst genau dieses Problem

Zählt man je Fenster nur die Komponenten, die **keine** Homologenreihen-Mitglieder sind — also das,
was nach einem Abzug der Polyolefin-Matrix übrig bliebe:

| Rezeptur | vor Subtraktion (med/p90/max) | nach Subtraktion (med/p90/max) | Fenster ohne Rest |
|---|---|---|---|
| `coelution_stress` | 7 / 8 / 11 | **3 / 4 / 7** | 10 von 31 |
| `pcr_mixed_polyolefin` | 7 / 9 / 12 | **3 / 4 / 8** | 7 von 31 |
| `trace_pet_in_polyolefin` | 7 / 8 / 9 | **3 / 4 / 5** | 11 von 29 |

Das ist der Unterschied zwischen „numerisch aussichtslos" und „Routine". Rund ein Drittel der
Fenster enthält danach überhaupt keinen Analyten mehr und kann übersprungen werden.

> **Konsequenz für den Plan:** Der ursprüngliche Auftrag nennt MCR-ALS vor der Matrix-Subtraktion.
> Die Reihenfolge wird umgedreht. Die Subtraktion ist kein Nachbearbeitungsschritt, sondern die
> Voraussetzung dafür, dass die Kurvenauflösung überhaupt ein lösbares Problem bekommt.

### Befund 3 — Rangschätzung ist das eigentliche Risiko, nicht der ALS-Kern

Für die fünf größten Fenster von `pcr_mixed_polyolefin`, wahre Komponentenzahl gegen drei gängige
Schätzer:

| wahr | SVD > 1 % σ₁ (roh) | SVD > 1 % σ₁ (vorverarbeitet) | Marchenko-Pastur-Kante |
|---|---|---|---|
| 12 | 8 | 7 | 108 |
| 9 | 8 | 8 | 109 |
| 9 | 8 | 8 | 108 |
| 9 | 32 | **173** | 129 |
| 8 | 7 | 7 | 94 |

Drei Dinge sind daran wichtig:

1. **Die Marchenko-Pastur-Kante ist unbrauchbar** (94–129 statt 8–12). Das Rauschen ist nicht
   i.i.d.: Schrotrauschen skaliert mit dem Signal, und die Baseline-Korrektur hinterlässt
   strukturierte Residuen. Die Annahme hinter MP ist schlicht verletzt.
2. **Der relative SVD-Schwellwert unterschätzt systematisch** (7–8 statt 8–12) — und das ist
   *korrekt*: benachbarte Homologe haben Spektren-Cosinus > 0,98 und tragen kaum unabhängigen
   Rang bei. Der **effektive chemische Rang liegt unter der wahren Komponentenzahl.**
3. **Ein Fenster bricht komplett aus** (wahr 9, geschätzt 173 nach Vorverarbeitung — schlechter als
   roh mit 32). Dort hinterlässt die Baseline-Korrektur offenbar strukturierte Residuen über viele
   Kanäle. Dieser Fall muss vor MS2.3 verstanden sein, sonst produziert die Rangschätzung
   gelegentlich Unsinn ohne Warnung.

> **Konsequenz für den Plan:** Punkt 2 bedeutet, dass „alle 191 Komponenten zurückgewinnen" kein
> sinnvolles Ziel ist. Das Ziel ist, **die Analyten zurückzugewinnen, die interessieren** — die
> Fremdpolymer- und Additivmarker — und die Homologenreihe als strukturiertes Ganzes zu behandeln
> statt als 120 Einzelkomponenten.

### Messskript reproduzieren

Die Zahlen entstehen aus `truth.coeluting_groups()`, `ComponentTruth.role.is_homologue` und einer
SVD je Fenster. Der Benchmark-Harness aus MS2.0 macht daraus einen festen Report, damit die Werte
bei jeder Änderung nachgeführt werden.

---

## Meilenstein 2 — Deconvolution-Engine

**Ziel:** Aus einem PCR-Pyrogramm die Spurenanalyten freilegen und ihre Reinspektren und Flächen
zurückgewinnen — quantifiziert gegen den Ground Truth aus MS1.

### MS2.0 — Bewertungs-Harness (zuerst, ohne ihn ist nichts messbar)

`src/pyrecycle_analytics/validation/`

```python
match_components(recovered_S, true_S) -> ComponentMatching
```
Zuordnung über die ungarische Methode auf der Cosinus-Ähnlichkeitsmatrix. Unverzichtbar, weil
MCR-ALS die Komponenten in beliebiger Reihenfolge zurückgibt — ohne Zuordnung ist jede Metrik
bedeutungslos.

```python
score_resolution(sample: SyntheticPyrogram, result: ResolutionResult) -> ResolutionScore
```

Kennzahlen: Spektren-Cosinus je zugeordneter Komponente, Profil-Korrelation, relativer
Flächenfehler, Lack of Fit (LOF %), erklärte Varianz R², Anzahl nicht zugeordneter wahrer
Komponenten (Misses) und erfundener Komponenten (False Positives).

**Wichtig:** Zusätzlich eine triviale Vergleichsbasis implementieren (Peak am TIC integrieren,
Spektrum am Apex nehmen). Ohne sie ist nicht belegbar, dass MCR-ALS überhaupt etwas verbessert.

*Akzeptanz:* Harness läuft über alle 10 Rezepturen und schreibt einen versionierbaren
Report; die Vergleichsbasis ist ausgewertet und dokumentiert.

### MS2.1 — Fensterbildung und Rangschätzung

`deconvolution/windows.py`, `deconvolution/rank.py`

- Fensterbildung auf dem vorverarbeiteten TIC plus kanalweiser Aktivität; Ziel ist eine
  Partitionierung, die `truth.coeluting_groups()` nahekommt, ohne den Truth zu kennen.
- Rangschätzung mit **mehreren** Schätzern und explizitem Konfidenzmaß:
  Malinowski-Indikatorfunktion (IND), Parallelanalyse per Permutation, sowie EFA
  (Evolving Factor Analysis — der chromatographie-klassische Ansatz, der die zeitliche
  Struktur nutzt statt sie wegzuwerfen).
- `estimate_rank(window) -> RankEstimate` gibt Einzelergebnisse **und** eine Warnung zurück, wenn
  die Schätzer auseinanderlaufen.

*Akzeptanz:* Auf den drei Benchmark-Rezepturen liegt der geschätzte Rang in ≥ 80 % der Fenster
innerhalb von ±1 des effektiven Rangs; das Ausreißer-Fenster aus Befund 3 ist erklärt und
entweder behoben oder erkannt und markiert. Fenstergrenzen decken ≥ 90 % der wahren Cluster ab.

### MS2.2 — Polyolefin-Matrixmodell und Subtraktion

`matrix/polyolefin.py` — **das wertvollste Einzelstück von MS2** (siehe Befund 2).

```python
subtract_polymer_matrix(cube, matrix_type="PE_PP_Backbone", ...) -> MatrixSubtractionResult
```

Ansatz: Die Homologenreihe ist kein beliebiger Untergrund, sondern hochstrukturiert — regelmäßige
Retentionsabstände (das Kováts-Modell steckt bereits in `core/peakshapes.py` und
`AlkaneRetentionModel`), bekannte Spektrenfamilien, glatte Amplitudenverteilung über die
Kettenlänge. Diese Struktur wird als parametrisches Modell angesetzt und per NNLS an die Daten
gefittet, statt einen generischen Untergrund zu schätzen.

Rückgabe: Residual-Cube, gefittetes Matrixmodell, Diagnostik (Anteil erklärter Varianz,
Verzweigungsgrad, Kettenlängenverteilung — Letztere ist bereits ein Zwischenergebnis für MS3).

*Akzeptanz:*
- Median-Komponentenzahl je Fenster im Residual ≤ 4 (gemessener Zielwert: 3).
- **Kein Analyt darf zerstört werden:** Die Fläche jedes Nicht-Matrix-Markers (PET, PA6, PVC,
  Additive) darf sich durch die Subtraktion um höchstens 10 % ändern. Das ist die schärfere und
  wichtigere Bedingung — ein Abzug, der die Matrix entfernt und dabei den Spurenanalyten
  mitnimmt, ist wertlos.
- Auf `trace_pet_in_polyolefin`: Signal-zu-Untergrund der PET-Marker um ≥ 5× verbessert.
- Residuum enthält keine systematische Reststruktur der Homologenreihe (Prüfung: Autokorrelation
  des Residual-TIC beim Kettenlängen-Abstand).

### MS2.3 — MCR-ALS auf dem Residuum

`deconvolution/mcrals.py`

- Initialisierung: SIMPLISMA (Reinvariablen-Detektion); EFA als Alternative.
- ALS-Schleife mit Nebenbedingungen: Nichtnegativität auf C und S, Unimodalität auf C (ein
  Chromatographie-Peak hat genau ein Maximum), Normierung von S.
- Konvergenz über LOF-Änderung; Rückgabe enthält `converged`, `iterations`, `lof`, `rank`.

*Vorgeschlagene Akzeptanz (nach Umsetzung zu validieren):*
- Spektren-Cosinus ≥ 0,95 für die diskreten Marker (Styrol, Caprolactam, Benzoesäure, DEHP) in
  `coelution_stress`.
- Flächenrückgewinnung ±15 % für Marker mit Auflösung R > 0,4 gegenüber dem nächsten Nachbarn.
- Auf `trace_pet_in_polyolefin`: PET-Marker-Fläche ±25 % — bei 0,2 % Signalanteil ist das
  realistisch, nicht bescheiden.
- Deutliche Verbesserung gegenüber der trivialen Vergleichsbasis aus MS2.0, sonst ist der Aufwand
  nicht gerechtfertigt.

Diese Schwellen sind bewusst als Vorschlag markiert. Sie stammen aus Erfahrungswerten, nicht aus
einer Messung — im Gegensatz zu allem in Abschnitt 0. Sie werden nach der ersten lauffähigen
Implementierung überprüft und gegebenenfalls begründet angepasst statt stillschweigend gesenkt.

### MS2.4 — Mehrlauf-Auswertung (optional, nach hinten priorisierbar)

`deconvolution/parafac2.py` — nutzt `generate_series()` aus MS1: gleiche Chemie, verschobene
Zeitachsen. PARAFAC2 existiert genau für diesen Fall. Alternativ ein leichteres RT-Alignment
(COW/PTW) vorschalten. Nur angehen, wenn MS2.1–2.3 stehen; für einen Einzellauf-Rezyklat-Pass
nicht erforderlich.

### Risiken MS2

| Risiko | Prüfung / Gegenmaßnahme |
|---|---|
| Rangschätzung bricht sporadisch aus (Befund 3) | Mehrere Schätzer, Divergenzwarnung, Ausreißerfenster vorab analysieren |
| Rotationsmehrdeutigkeit trotz Nebenbedingungen | Gegen Truth messen statt gegen LOF; LOF allein ist kein Qualitätsnachweis |
| Subtraktion entfernt Analytsignal | Harte Akzeptanzbedingung (≤ 10 % Flächenänderung), pro Marker geprüft |
| Rechenzeit über ~30 Fenster × 10 Rezepturen | Fenster sind unabhängig → parallelisierbar; Profiling vor Optimierung |

---

## Meilenstein 3 — Marker-Bibliothek und Degradations-Index

**Ziel:** Aus den zurückgewonnenen Spektren und Flächen auf Polymere und Alterungszustand
schließen.

### MS3.1 — Relationales Schema

`src/pyrecycle_analytics/library/` mit SQLAlchemy 2.0, SQLite lokal / PostgreSQL produktiv,
Alembic-Migrationen.

Kern der Tabellen:

- `polymer` — Klasse, ISO-Kürzel, Bezeichnung
- `compound` — Name, CAS, Summenformel, Molmasse
- `reference_spectrum` / `spectrum_peak` — m/z und relative Intensität, mit Quellenangabe
- `retention_index` — Compound, stationäre Phase, RI-Wert, Methode
- `marker_pattern` — z. B. „PS Styrol-Triade", verknüpft mit einem Polymer
- `marker_pattern_member` — Compound, Rolle (`MarkerRole` aus MS1), erwartetes
  Mengenverhältnis, Toleranz
- `response_factor` — Polymer, Pyrolysetemperatur, Faktor (die Grundlage aus MS1 wird persistiert)
- `measurement` / `identification` — Ergebnisse mit Rückverweis auf Prüfsumme und
  Verarbeitungskette aus MS1

### MS3.2 — Retentionsindex-Verankerung

Retentionszeiten driften, Retentionsindizes nicht. In einem polyolefinreichen Rezyklat liefert die
**Matrix selbst die Kováts-Leiter**: die n-Alkan-Reihe ist immer vorhanden und wird in MS2.2
ohnehin schon modelliert und mit Kettenlängen versehen.

Damit wird die Identifikation gerätesäulen- und methodenunabhängig, ohne dass ein separater
Standard eingespritzt werden muss. Das ist ein direkter Nebennutzen der Matrix-Subtraktion und
sollte explizit als Schnittstelle zwischen MS2.2 und MS3 gebaut werden.

### MS3.3 — Identifikationslogik

Bewertung je Kandidat aus drei Faktoren: Spektrenähnlichkeit (gewichtetes Skalarprodukt nach
NIST-Art, das hohe m/z stärker gewichtet), RI-Übereinstimmung, und Konsistenz der
Marker-Triaden-Verhältnisse. Ergebnis ist eine Kennzahl mit Konfidenz, kein Ja/Nein.

**Methodisches Risiko, das aktiv adressiert werden muss:** Die Bibliothek darf **nicht** aus
`tests/reference_spectra.py` abgeleitet werden. Sonst validiert sich die Identifikation gegen ihre
eigene Quelle und jede Kennzahl ist wertlos. Konkrete Maßnahme:

- Importpfad für NIST-MSP-Dateien bzw. eine kuratierte CSV mit Literaturquelle je Eintrag.
- Ein Test, der aktiv prüft, dass Bibliotheks- und Testtabelle **nicht** identisch sind.
- Der Erfolg auf dem synthetischen Benchmark wird dadurch schlechter aussehen als er könnte — das
  ist beabsichtigt und das einzige ehrliche Vorgehen.

### MS3.4 — DegradationEngine

`degradation/engine.py` — konkrete Kennzahlen, alle gegen eine Virgin-Referenz normiert
(absolute Werte sind ohne Referenz nicht interpretierbar):

| Kennzahl | Definition | Basis in MS1 |
|---|---|---|
| Carbonyl-Index | Σ(Ketone + Aldehyde + Säuren) / Σ(n-Alkane) | Rezeptur `aged_hdpe`: 10× über Virgin |
| Alken/Alkan-Verhältnis | je Kettenlänge, gemittelt | im Generator konstant 0,62 für virgin PE |
| Verzweigungsindex | Σ(iso-Alkene) / Σ(n-Alkane) | LDPE > 3× HDPE, als Test abgesichert |
| Kettenlängen-Verschiebung | flächengewichtetes mittleres C der Alkane | sinkt mit Alterung |
| Oligomer-Schiefe (Styrolics) | Monomer / Trimer | steigt > 2× bei starker Alterung |
| Säureanteil an der Oxidation | Säuren / Σ(Oxidationsprodukte) | steigt monoton mit Alterungsgrad |

Die letzte Kennzahl trennt *mild* von *stark* gealtert, während der Carbonyl-Index nur sagt, *dass*
oxidiert wurde. Der Generator bildet diese Ordnung bereits ab und sichert sie per Test.

*Akzeptanz MS3:* Auf den Rezepturen mit bekanntem Alterungsgrad korreliert der Degradations-Index
monoton mit `truth.degradation_levels` (Spearman ρ ≥ 0,9 über eine Serie von Alterungsgraden);
LDPE und HDPE werden über den Verzweigungsindex getrennt; die Polymer-Identifikation erreicht auf
`pcr_mixed_polyolefin` alle Fraktionen ≥ 1 % Massenanteil ohne False Positive.

---

## Meilenstein 4 — Rezyklat-Pass und Reporting-API

**Ziel:** Ein strukturierter, prüffähiger Digitaler Rezyklat-Pass als JSON und PDF.

### MS4.1 — FastAPI

```
POST /samples                     Rohdatei hochladen  (nutzt read_pyrogram_bytes aus MS1)
GET  /samples/{id}                Status, Metadaten, Reader-Warnungen
POST /samples/{id}/analyse        Pipeline mit Konfiguration anstoßen (asynchron)
GET  /samples/{id}/passport       Rezyklat-Pass als JSON
GET  /samples/{id}/passport.pdf   dito als PDF
GET  /library/polymers|markers    Bibliotheksinhalt
```

Lange Läufe asynchron mit Job-Status; die Analyse eines Vollauflösungs-Pyrogramms ist keine
Request-Latenz.

### MS4.2 — Passport-Schema

`data_schemas/passport.py`, Pydantic, versioniert (`schema_version`), damit ältere Pässe lesbar
bleiben.

Inhalt: polymere Hauptfraktionen in %, detektierte Schadstoffe und REACH-Marker, Degradations- und
Verzweigungsindex — dazu zwingend die vollständige Provenienz aus MS1 (Prüfsumme,
Verarbeitungskette, Bibliotheksversion, Methodenbeschreibung) und **je Angabe eine Unsicherheit
und ein Konfidenzniveau**.

### MS4.3 — Zwei Punkte, die im Pass ehrlich benannt werden müssen

**Prozentangaben sind ohne Kalibrierung semiquantitativ.** Py-GC/MS liefert Signalanteile. Der
Response-Faktor-Schritt aus MS1 korrigiert den systematischen Teil (1 % PVC ergibt 0,3 % Signal),
aber die absolute Skala braucht gravimetrisch angesetzte Referenzmischungen. Der Pass muss den
Kalibrierstatus ausweisen — mit Kalibrierung „%", ohne „% (semiquantitativ, unkalibriert)". Eine
unkalibrierte Prozentzahl, die wie eine kalibrierte aussieht, ist in einem Dokument mit dem Wort
„Pass" im Namen ein echtes Problem.

**REACH-Grenzwerte sind Massenanteile.** Ein „detektiert"-Flag für DEHP, DBP, BBP, DIBP, TBBPA,
SCCP oder Bisphenol A ist gut belegbar. Die Aussage „0,08 % und damit unter 0,1 %" ist es ohne
Kalibrierstandards nicht. Der Regulatorik-Anhang führt Substanz, Rechtsgrundlage, Grenzwert und
den **tatsächlich erreichten Nachweisstatus** getrennt auf.

### MS4.4 — PDF

WeasyPrint (HTML/CSS-Template, wartbar) gegenüber ReportLab bevorzugt. Deterministische Ausgabe,
damit zwei Läufe derselben Probe byte-identische Pässe erzeugen und Diffs aussagekräftig bleiben.

*Akzeptanz MS4:* End-to-End-Test von Rohdatei bis PDF; Schema-Round-Trip; Pass einer
`lab-blend`-Rezeptur gibt die bekannte Zusammensetzung innerhalb der ausgewiesenen Unsicherheit
wieder; unkalibrierte Läufe sind im Dokument sichtbar als solche markiert.

---

## Querschnitt

- **Benchmark als Regressionsschutz:** Der Harness aus MS2.0 läuft in CI und schreibt seine
  Kennzahlen fort. Eine Änderung, die den Spektren-Cosinus senkt, fällt auf, bevor sie gemerged
  wird.
- **Rechenzeit:** Fenster sind unabhängig — Parallelisierung ist der erste Hebel. Vor Optimierung
  profilieren.
- **Persistenz:** Ergebnisse gehören in die MS3-Datenbank, nicht in Dateien neben dem Rohdatensatz.
- **Streamlit:** Die Oberfläche aus MS1 wächst je Meilenstein mit — aufgelöste Profile in MS2,
  Identifikationstabelle in MS3, Pass-Vorschau in MS4.

## Reihenfolge und Abhängigkeiten

```
MS2.0 Harness ──► MS2.1 Fenster/Rang ──► MS2.2 Matrixmodell ──► MS2.3 MCR-ALS ──► (MS2.4 PARAFAC2)
                                                │                      │
                                                ▼                      ▼
                                      MS3.2 RI-Verankerung      MS3.3 Identifikation
                                                                       │
                                      MS3.1 Schema ────────────────────┤
                                      MS3.4 Degradation ───────────────┤
                                                                       ▼
                                                              MS4 Rezyklat-Pass
```

MS2.2 ist der kritische Pfad: Es senkt die Fensterkomplexität für MS2.3 **und** liefert die
Kováts-Leiter für MS3.2 **und** die Kettenlängen- und Verzweigungsstatistik für MS3.4. Wenn nur ein
Arbeitspaket aus MS2 fertig wird, dann dieses.
