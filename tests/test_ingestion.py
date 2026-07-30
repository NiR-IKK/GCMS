"""Tests for the raw-data ingestion layer.

The strategy throughout is round-tripping against the *real* parsers rather than
mocks: a synthetic cube is written to a genuine mzML or ANDI-MS netCDF file and
read back with pyopenms / scipy. That is the only way to catch the failures that
actually happen with instrument files — wrong scan-index arithmetic, minutes
mistaken for seconds, ions silently falling outside the m/z grid.

Malformed files are constructed deliberately, because the ingestion layer's job is
to fail loudly on bad data rather than quietly produce a plausible-looking cube.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

from data_schemas.acquisition import GcConditions, PyrolysisConditions, SampleMetadata
from data_schemas.enums import PolymerClass, RecyclateStream, SourceFormat
from pyrecycle_analytics.exceptions import (
    CorruptRawDataError,
    MissingDependencyError,
    UnsupportedFormatError,
)
from pyrecycle_analytics.ingestion import (
    IngestOptions,
    detect_format,
    read_andi_cdf,
    read_pyrogram,
    sha256_of_file,
    write_andi_cdf,
    write_mzml,
)
from tests.conftest import requires_pyopenms
from tests.synthetic_data import SyntheticPyrogram


@pytest.fixture(scope="module")
def cdf_path(tmp_path_factory, small_io_sample: SyntheticPyrogram) -> Path:  # noqa: ANN001
    """A real ANDI-MS netCDF file written from the synthetic cube."""
    directory = tmp_path_factory.mktemp("andi")
    return write_andi_cdf(small_io_sample.cube, directory / "sample.cdf")


@pytest.fixture(scope="module")
def mzml_path(tmp_path_factory, small_io_sample: SyntheticPyrogram) -> Path:  # noqa: ANN001
    """A real mzML file written from the synthetic cube."""
    directory = tmp_path_factory.mktemp("mzml")
    return write_mzml(small_io_sample.cube, directory / "sample.mzML")


def _full_range(sample: SyntheticPyrogram) -> tuple[float, float]:
    """The cube's exact m/z window, so the round trip is not truncated."""
    return (float(sample.cube.mz_axis[0]), float(sample.cube.mz_axis[-1]))


class TestFormatDetection:
    def test_detects_andi_netcdf_by_magic_bytes(self, cdf_path: Path) -> None:
        assert detect_format(cdf_path) is SourceFormat.ANDI_CDF

    @requires_pyopenms
    def test_detects_mzml_by_root_element(self, mzml_path: Path) -> None:
        assert detect_format(mzml_path) is SourceFormat.MZML

    def test_content_beats_a_misleading_extension(
        self, tmp_path: Path, cdf_path: Path
    ) -> None:
        """A netCDF file named ``.mzML`` must still be recognised as netCDF.

        Analysts rename files; trusting the extension would send the wrong parser
        at it and produce a confusing XML error instead of a working import.
        """
        renamed = tmp_path / "mislabelled.mzML"
        renamed.write_bytes(cdf_path.read_bytes())
        assert detect_format(renamed) is SourceFormat.ANDI_CDF

    def test_hdf5_container_is_rejected_with_actionable_advice(
        self, tmp_path: Path
    ) -> None:
        """netCDF-4 wears a ``.cdf`` extension but is HDF5 and cannot be parsed.

        This is a common real-world stumbling block, so the error names the
        remedy rather than surfacing a low-level parse failure.
        """
        path = tmp_path / "netcdf4.cdf"
        path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
        with pytest.raises(UnsupportedFormatError, match="HDF5 container"):
            detect_format(path)

    def test_unknown_content_is_rejected_and_lists_what_is_supported(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("retention time, area\n1.0, 2.0\n")
        with pytest.raises(UnsupportedFormatError, match="cannot identify the format"):
            detect_format(path)

    def test_extension_is_used_when_content_is_inconclusive(self, tmp_path: Path) -> None:
        path = tmp_path / "truncated.mzXML"
        path.write_bytes(b"\x00\x01\x02\x03")
        assert detect_format(path) is SourceFormat.MZXML

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no such file"):
            detect_format(tmp_path / "absent.mzML")

    def test_directory_is_not_mistaken_for_a_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            detect_format(tmp_path)


class TestAndiRoundTrip:
    def test_axes_and_intensities_survive_the_round_trip(
        self, cdf_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        restored = read_pyrogram(cdf_path, mz_range=_full_range(small_io_sample))
        original = small_io_sample.cube

        assert restored.shape == original.shape
        assert np.allclose(restored.retention_times, original.retention_times, atol=1e-3)
        assert np.array_equal(restored.mz_axis, original.mz_axis)
        # ANDI stores intensities as float32, so exact equality is not expected.
        assert np.allclose(restored.intensities, original.intensities, rtol=1e-5, atol=1e-2)

    def test_total_ion_current_is_conserved(
        self, cdf_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        """Quantification is relative to the TIC, so it must survive I/O."""
        restored = read_pyrogram(cdf_path, mz_range=_full_range(small_io_sample))
        assert restored.total_signal == pytest.approx(
            small_io_sample.cube.total_signal, rel=1e-5
        )

    def test_andi_global_attributes_are_preserved(self, cdf_path: Path) -> None:
        restored = read_andi_cdf(cdf_path)
        assert restored.metadata.extra["experiment_type"] == "Centroided Mass Spectrum"
        assert restored.metadata.extra["dataset_origin"] == "PyRecycle-Analytics"
        assert "ms_template_revision" in restored.metadata.extra

    def test_ms_conditions_are_recovered_from_the_attributes(self, cdf_path: Path) -> None:
        ms = read_andi_cdf(cdf_path).metadata.acquisition.ms
        assert ms.ionisation.value == "EI"
        assert ms.polarity.value == "positive"
        assert ms.acquisition_mode.value == "full-scan"

    def test_source_provenance_is_recorded(self, cdf_path: Path) -> None:
        restored = read_andi_cdf(cdf_path)
        assert restored.metadata.source_format is SourceFormat.ANDI_CDF
        assert restored.metadata.source_path == cdf_path
        assert restored.metadata.source_checksum == sha256_of_file(cdf_path)

    def test_scan_rate_is_derived_from_the_retention_axis(
        self, cdf_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        restored = read_andi_cdf(cdf_path)
        expected = 1.0 / small_io_sample.cube.mean_scan_period_s
        assert restored.metadata.acquisition.ms.scan_rate_hz == pytest.approx(
            expected, rel=1e-3
        )


@requires_pyopenms
class TestMzmlRoundTrip:
    def test_axes_and_intensities_survive_the_round_trip(
        self, mzml_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        restored = read_pyrogram(mzml_path, mz_range=_full_range(small_io_sample))
        original = small_io_sample.cube

        assert restored.shape == original.shape
        assert np.allclose(restored.retention_times, original.retention_times, atol=1e-3)
        assert np.array_equal(restored.mz_axis, original.mz_axis)
        assert np.allclose(restored.intensities, original.intensities, rtol=1e-6, atol=1e-6)

    def test_total_ion_current_is_conserved(
        self, mzml_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        restored = read_pyrogram(mzml_path, mz_range=_full_range(small_io_sample))
        assert restored.total_signal == pytest.approx(
            small_io_sample.cube.total_signal, rel=1e-9
        )

    def test_source_provenance_is_recorded(self, mzml_path: Path) -> None:
        restored = read_pyrogram(mzml_path)
        assert restored.metadata.source_format is SourceFormat.MZML
        assert restored.metadata.source_checksum == sha256_of_file(mzml_path)

    def test_only_ms1_spectra_are_ingested(self, tmp_path: Path) -> None:
        """Py-GC/MS is MS1; an MS2 scan in the file must not enter the cube."""
        import pyopenms as oms

        experiment = oms.MSExperiment()
        for index in range(10):
            spectrum = oms.MSSpectrum()
            spectrum.setRT(float(index))
            spectrum.setMSLevel(1)
            spectrum.set_peaks((np.array([55.0, 57.0]), np.array([100.0, 200.0])))
            experiment.addSpectrum(spectrum)

            fragment = oms.MSSpectrum()
            fragment.setRT(float(index) + 0.5)
            fragment.setMSLevel(2)
            fragment.set_peaks((np.array([41.0]), np.array([9999.0])))
            experiment.addSpectrum(fragment)

        path = tmp_path / "mixed-levels.mzML"
        handler = oms.MzMLFile()
        handler.setLogType(oms.LogType.NONE)
        handler.store(str(path), experiment)

        cube = read_pyrogram(path)
        assert cube.n_scans == 10
        assert cube.total_signal == pytest.approx(10 * 300.0)
        assert not any("MS level" in warning for warning in cube.metadata.reader_warnings)


@requires_pyopenms
class TestOtherXmlFormats:
    def test_mzxml_is_detected_and_read(self, tmp_path: Path) -> None:
        """mzXML still turns up from older converters and has to work."""
        import pyopenms as oms

        experiment = oms.MSExperiment()
        for index in range(8):
            spectrum = oms.MSSpectrum()
            spectrum.setRT(60.0 + 2.0 * index)
            spectrum.setMSLevel(1)
            spectrum.set_peaks((np.array([55.0, 57.0, 71.0]), np.array([10.0, 20.0, 5.0])))
            experiment.addSpectrum(spectrum)

        path = tmp_path / "legacy.mzXML"
        handler = oms.MzXMLFile()
        handler.setLogType(oms.LogType.NONE)
        handler.store(str(path), experiment)

        assert detect_format(path) is SourceFormat.MZXML
        cube = read_pyrogram(path)
        assert cube.n_scans == 8
        assert cube.metadata.source_format is SourceFormat.MZXML
        assert cube.total_signal == pytest.approx(8 * 35.0)

    def test_missing_retention_times_are_rejected_with_a_clear_message(
        self, tmp_path: Path
    ) -> None:
        """A '-1' retention-time placeholder must not reach the schema layer.

        Some converters — including OpenMS's own mzData handler — emit -1 when no
        scan time was recorded. A pyrogram without a time axis is uninterpretable,
        so the reader says exactly that instead of leaking a validation error about
        an internal field name.
        """
        import pyopenms as oms

        experiment = oms.MSExperiment()
        for _ in range(4):
            spectrum = oms.MSSpectrum()
            spectrum.setMSLevel(1)  # RT deliberately left unset
            spectrum.set_peaks((np.array([55.0]), np.array([10.0])))
            experiment.addSpectrum(spectrum)

        path = tmp_path / "no-times.mzData"
        handler = oms.MzDataFile()
        handler.setLogType(oms.LogType.NONE)
        handler.store(str(path), experiment)

        with pytest.raises(CorruptRawDataError, match="negative retention time"):
            read_pyrogram(path)

    def test_unparseable_xml_is_reported_as_corrupt(self, tmp_path: Path) -> None:
        path = tmp_path / "truncated.mzML"
        path.write_text('<?xml version="1.0"?>\n<mzML><run><spectrumList>')
        with pytest.raises(CorruptRawDataError, match="could not parse"):
            read_pyrogram(path)

    def test_pyopenms_is_reported_as_available(self) -> None:
        from pyrecycle_analytics.ingestion import pyopenms_available

        assert pyopenms_available() is True


class TestReadOptions:
    def test_rt_range_trims_the_run(
        self, cdf_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        """Cutting the solvent front and the terminal bleed ramp is routine."""
        start, end = 150.0, 300.0
        restored = read_pyrogram(cdf_path, rt_range_s=(start, end))
        assert restored.n_scans < small_io_sample.cube.n_scans
        assert restored.retention_times[0] >= start
        assert restored.retention_times[-1] <= end

    def test_mz_range_restricts_the_grid_and_drops_outside_ions(
        self, cdf_path: Path
    ) -> None:
        restored = read_pyrogram(cdf_path, mz_range=(50.0, 100.0))
        assert restored.mz_axis[0] == 50.0
        assert restored.mz_axis[-1] == 100.0
        full = read_pyrogram(cdf_path)
        assert restored.total_signal < full.total_signal

    def test_max_scans_truncates_and_warns(self, cdf_path: Path) -> None:
        restored = read_pyrogram(cdf_path, max_scans=25)
        assert restored.n_scans == 25
        assert any("truncated" in warning for warning in restored.metadata.reader_warnings)

    def test_checksum_can_be_skipped(self, cdf_path: Path) -> None:
        assert read_pyrogram(cdf_path, compute_checksum=False).metadata.source_checksum is None

    def test_sample_metadata_can_be_supplied(self, cdf_path: Path) -> None:
        """Ingestion has to accept LIMS identity; the file cannot carry it."""
        sample = SampleMetadata(
            sample_id="PCR-LDPE-2024-07",
            stream=RecyclateStream.PCR_LDPE,
            batch="B-118",
            nominal_composition={str(PolymerClass.PE_LD): 0.95},
        )
        restored = read_pyrogram(cdf_path, options=IngestOptions(sample=sample))
        assert restored.metadata.sample.sample_id == "PCR-LDPE-2024-07"
        assert restored.metadata.sample.stream is RecyclateStream.PCR_LDPE

    def test_sample_id_defaults_to_the_file_stem(self, cdf_path: Path) -> None:
        assert read_pyrogram(cdf_path).metadata.sample.sample_id == "sample"

    def test_pyrolysis_and_gc_conditions_can_be_supplied(self, cdf_path: Path) -> None:
        """Marker ratios are only comparable within one method, so it is recorded."""
        options = IngestOptions(
            pyrolysis=PyrolysisConditions(temperature_c=550.0, duration_s=6.0),
            gc=GcConditions(column_name="Ultra ALLOY+-5", split_ratio=100.0),
        )
        acquisition = read_pyrogram(cdf_path, options=options).metadata.acquisition
        assert acquisition.pyrolysis.temperature_c == 550.0
        assert acquisition.gc.column_name == "Ultra ALLOY+-5"

    def test_unknown_option_name_is_rejected_with_a_helpful_message(
        self, cdf_path: Path
    ) -> None:
        with pytest.raises(TypeError, match="unknown ingest option"):
            read_pyrogram(cdf_path, mz_rnage=(50.0, 100.0))

    def test_rt_range_selecting_nothing_is_an_error(self, cdf_path: Path) -> None:
        with pytest.raises(CorruptRawDataError, match="selects none of the"):
            read_pyrogram(cdf_path, rt_range_s=(1e6, 2e6))

    def test_mz_range_outside_the_data_warns_about_an_empty_matrix(
        self, cdf_path: Path
    ) -> None:
        restored = read_pyrogram(cdf_path, mz_range=(900.0, 1000.0))
        assert not restored.intensities.any()
        assert any("all zeros" in warning for warning in restored.metadata.reader_warnings)


class TestMalformedAndiFiles:
    def _write(self, path: Path, **variables: np.ndarray) -> Path:
        """Write a netCDF-3 file containing exactly the given variables."""
        with netcdf_file(str(path), "w") as dataset:
            dimensions: dict[str, int] = {}
            for name, values in variables.items():
                dimension = f"dim_{name}"
                dimensions[dimension] = values.size
                dataset.createDimension(dimension, values.size)
                kind = "d" if values.dtype.kind == "f" else "i"
                variable = dataset.createVariable(name, kind, (dimension,))
                variable[:] = values
        return path

    def test_missing_required_variables_are_named_in_the_error(
        self, tmp_path: Path
    ) -> None:
        path = self._write(
            tmp_path / "incomplete.cdf", scan_acquisition_time=np.arange(5.0)
        )
        with pytest.raises(CorruptRawDataError, match="mass_values, intensity_values"):
            read_andi_cdf(path)

    def test_mismatched_point_arrays_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "ragged.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 2)
            dataset.createDimension("point_number", 4)
            dataset.createDimension("short_number", 3)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                1.0,
                2.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = [
                50.0,
                51.0,
                52.0,
                53.0,
            ]
            dataset.createVariable("intensity_values", "d", ("short_number",))[:] = [
                1.0,
                2.0,
                3.0,
            ]
        with pytest.raises(CorruptRawDataError, match="4 entries but 'intensity_values'"):
            read_andi_cdf(path)

    def test_scan_index_of_the_wrong_length_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad-index.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 3)
            dataset.createDimension("point_number", 6)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                1.0,
                2.0,
                3.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = np.arange(
                50.0, 56.0
            )
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = np.ones(6)
            dataset.createDimension("wrong", 2)
            dataset.createVariable("scan_index", "i", ("wrong",))[:] = [0, 3]
        with pytest.raises(CorruptRawDataError, match="'scan_index' has 2 entries"):
            read_andi_cdf(path)

    def test_scan_index_running_past_the_point_list_is_clipped_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        """Truncated acquisitions really do produce this; clip rather than crash."""
        path = tmp_path / "overflow.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 2)
            dataset.createDimension("point_number", 4)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                1.0,
                2.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = [
                50.0,
                51.0,
                52.0,
                53.0,
            ]
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = np.ones(4)
            dataset.createVariable("scan_index", "i", ("scan_number",))[:] = [0, 2]
            dataset.createVariable("point_count", "i", ("scan_number",))[:] = [2, 99]

        cube = read_andi_cdf(path)
        assert cube.n_scans == 2
        assert any("index past the end" in w for w in cube.metadata.reader_warnings)

    def test_missing_point_count_is_derived_from_scan_index(self, tmp_path: Path) -> None:
        path = tmp_path / "no-count.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 3)
            dataset.createDimension("point_number", 6)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                1.0,
                2.0,
                3.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = [
                50.0,
                51.0,
                50.0,
                52.0,
                51.0,
                53.0,
            ]
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ]
            dataset.createVariable("scan_index", "i", ("scan_number",))[:] = [0, 2, 4]

        cube = read_andi_cdf(path)
        assert cube.n_scans == 3
        assert cube.total_signal == pytest.approx(21.0)

    def test_rectangular_file_without_index_or_count_is_inferred(
        self, tmp_path: Path
    ) -> None:
        """Some minimal exports omit both; equal-length scans are then assumed."""
        path = tmp_path / "rectangular.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 3)
            dataset.createDimension("point_number", 6)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                10.0,
                20.0,
                30.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = [
                50.0,
                51.0,
                50.0,
                51.0,
                50.0,
                51.0,
            ]
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ]

        cube = read_andi_cdf(path)
        assert cube.n_scans == 3
        assert cube.total_signal == pytest.approx(21.0)
        assert any("2 points per scan" in w for w in cube.metadata.reader_warnings)

    def test_non_rectangular_file_without_index_or_count_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """Guessing an uneven split would silently misalign every spectrum."""
        path = tmp_path / "indivisible.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 4)
            dataset.createDimension("point_number", 6)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                10.0,
                20.0,
                30.0,
                40.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = np.arange(
                50.0, 56.0
            )
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = np.ones(6)

        with pytest.raises(CorruptRawDataError, match="not divisible by the scan count"):
            read_andi_cdf(path)

    def test_negative_offsets_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "negative.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 2)
            dataset.createDimension("point_number", 4)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                10.0,
                20.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = np.arange(
                50.0, 54.0
            )
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = np.ones(4)
            dataset.createVariable("scan_index", "i", ("scan_number",))[:] = [0, -2]
            dataset.createVariable("point_count", "i", ("scan_number",))[:] = [2, 2]

        with pytest.raises(CorruptRawDataError, match="negative values"):
            read_andi_cdf(path)

    def test_negative_retention_times_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "negative-time.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 2)
            dataset.createDimension("point_number", 2)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                -1.0,
                20.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = [50.0, 51.0]
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = np.ones(2)
            dataset.createVariable("scan_index", "i", ("scan_number",))[:] = [0, 1]
            dataset.createVariable("point_count", "i", ("scan_number",))[:] = [1, 1]

        with pytest.raises(CorruptRawDataError, match="negative retention time"):
            read_andi_cdf(path)

    def test_unrecognised_time_unit_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "odd-units.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 2)
            dataset.createDimension("point_number", 2)
            times = dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))
            times[:] = [10.0, 20.0]
            times.units = b"Furlongs"
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = [50.0, 51.0]
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = np.ones(2)
            dataset.createVariable("scan_index", "i", ("scan_number",))[:] = [0, 1]
            dataset.createVariable("point_count", "i", ("scan_number",))[:] = [1, 1]

        cube = read_andi_cdf(path)
        assert any("unrecognised time unit" in w for w in cube.metadata.reader_warnings)

    def test_all_empty_scans_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "all-empty.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 2)
            dataset.createDimension("point_number", 1)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                10.0,
                20.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = [50.0]
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = [1.0]
            dataset.createVariable("scan_index", "i", ("scan_number",))[:] = [0, 0]
            dataset.createVariable("point_count", "i", ("scan_number",))[:] = [0, 0]

        with pytest.raises(CorruptRawDataError, match="nothing to bin"):
            read_andi_cdf(path)

    def test_minutes_are_converted_to_seconds(self, tmp_path: Path) -> None:
        """Some vendors write minutes; using them raw would compress the run 60x."""
        path = tmp_path / "minutes.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 3)
            dataset.createDimension("point_number", 3)
            times = dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))
            times[:] = [1.0, 2.0, 3.0]
            times.units = b"Minutes"
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = [
                50.0,
                51.0,
                52.0,
            ]
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = np.ones(3)
            dataset.createVariable("scan_index", "i", ("scan_number",))[:] = [0, 1, 2]
            dataset.createVariable("point_count", "i", ("scan_number",))[:] = [1, 1, 1]

        cube = read_andi_cdf(path)
        assert np.allclose(cube.retention_times, [60.0, 120.0, 180.0])
        assert any("converted to seconds" in w for w in cube.metadata.reader_warnings)

    def test_unsorted_scans_are_reordered_with_a_warning(self, tmp_path: Path) -> None:
        path = tmp_path / "unsorted.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 3)
            dataset.createDimension("point_number", 3)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                30.0,
                10.0,
                20.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = [
                52.0,
                50.0,
                51.0,
            ]
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = [
                3.0,
                1.0,
                2.0,
            ]
            dataset.createVariable("scan_index", "i", ("scan_number",))[:] = [0, 1, 2]
            dataset.createVariable("point_count", "i", ("scan_number",))[:] = [1, 1, 1]

        cube = read_andi_cdf(path)
        assert np.allclose(cube.retention_times, [10.0, 20.0, 30.0])
        # The spectra must travel with their retention times, not stay in place.
        assert cube.intensities[0, cube.mz_index(50.0)] == pytest.approx(1.0)
        assert cube.intensities[2, cube.mz_index(52.0)] == pytest.approx(3.0)
        assert any("not sorted" in w for w in cube.metadata.reader_warnings)

    def test_duplicate_retention_times_are_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "duplicates.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 3)
            dataset.createDimension("point_number", 3)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                10.0,
                10.0,
                20.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = [
                50.0,
                50.0,
                51.0,
            ]
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = np.ones(3)
            dataset.createVariable("scan_index", "i", ("scan_number",))[:] = [0, 1, 2]
            dataset.createVariable("point_count", "i", ("scan_number",))[:] = [1, 1, 1]

        cube = read_andi_cdf(path)
        assert cube.n_scans == 2
        assert any("duplicate retention times" in w for w in cube.metadata.reader_warnings)

    def test_intensity_scale_factor_is_applied(self, tmp_path: Path) -> None:
        """ANDI allows scaled integer intensities; ignoring the factor skews ratios."""
        path = tmp_path / "scaled.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 2)
            dataset.createDimension("point_number", 2)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                10.0,
                20.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = [50.0, 51.0]
            intensities = dataset.createVariable("intensity_values", "d", ("point_number",))
            intensities[:] = [10.0, 20.0]
            intensities.scale_factor = 4.0
            dataset.createVariable("scan_index", "i", ("scan_number",))[:] = [0, 1]
            dataset.createVariable("point_count", "i", ("scan_number",))[:] = [1, 1]

        cube = read_andi_cdf(path)
        assert cube.total_signal == pytest.approx(120.0)

    def test_declared_tic_mismatch_is_reported(
        self, cdf_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        """The file's own TIC is an independent check on our binning.

        A narrowed m/z window legitimately discards ions, and the reader says so
        rather than silently reporting a smaller total than the instrument did.
        """
        restored = read_pyrogram(cdf_path, mz_range=(50.0, 80.0))
        assert any(
            "deviates from the file's 'total_intensity'" in warning
            for warning in restored.metadata.reader_warnings
        )

    def test_empty_scans_are_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "some-empty.cdf"
        with netcdf_file(str(path), "w") as dataset:
            dataset.createDimension("scan_number", 3)
            dataset.createDimension("point_number", 2)
            dataset.createVariable("scan_acquisition_time", "d", ("scan_number",))[:] = [
                10.0,
                20.0,
                30.0,
            ]
            dataset.createVariable("mass_values", "d", ("point_number",))[:] = [50.0, 51.0]
            dataset.createVariable("intensity_values", "d", ("point_number",))[:] = [1.0, 2.0]
            dataset.createVariable("scan_index", "i", ("scan_number",))[:] = [0, 1, 2]
            dataset.createVariable("point_count", "i", ("scan_number",))[:] = [1, 1, 0]

        cube = read_andi_cdf(path)
        assert any("contain no ions" in w for w in cube.metadata.reader_warnings)


class TestWriters:
    def test_extension_is_appended_when_missing(
        self, tmp_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        assert write_andi_cdf(small_io_sample.cube, tmp_path / "no-suffix").suffix == ".cdf"

    def test_sparse_output_omits_empty_channels(
        self, tmp_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        """Empty channels are dropped, which is how centroid data is stored.

        The check uses the noise-free cube: once detector noise is added, every
        channel of every scan carries a count, so there is nothing to omit — a
        measured cube is only sparse because the instrument thresholds it.
        """
        clean = small_io_sample.clean_cube
        assert not clean.intensities.all(), "fixture must contain empty channels"

        sparse = write_andi_cdf(clean, tmp_path / "sparse.cdf")
        dense = write_andi_cdf(clean, tmp_path / "dense.cdf", keep_zeros=True)
        assert sparse.stat().st_size < 0.7 * dense.stat().st_size

    def test_sparse_output_is_lossless(
        self, tmp_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        """Dropping zeros must not change any non-zero value."""
        clean = small_io_sample.clean_cube
        path = write_andi_cdf(clean, tmp_path / "lossless.cdf")
        restored = read_pyrogram(path, mz_range=_full_range(small_io_sample))
        assert np.allclose(restored.intensities, clean.intensities, rtol=1e-5, atol=1e-2)

    def test_dense_output_round_trips_identically(
        self, tmp_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        path = write_andi_cdf(small_io_sample.cube, tmp_path / "dense.cdf", keep_zeros=True)
        restored = read_pyrogram(path, mz_range=_full_range(small_io_sample))
        assert np.allclose(
            restored.intensities, small_io_sample.cube.intensities, rtol=1e-5, atol=1e-2
        )

    def test_written_tic_matches_the_cube(
        self, tmp_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        path = write_andi_cdf(small_io_sample.cube, tmp_path / "tic.cdf")
        with netcdf_file(str(path), "r", mmap=False) as dataset:
            declared = np.array(dataset.variables["total_intensity"][:])
        assert np.allclose(declared, small_io_sample.cube.tic, rtol=1e-9)

    @requires_pyopenms
    def test_mzml_writer_appends_the_conventional_extension(
        self, tmp_path: Path, small_io_sample: SyntheticPyrogram
    ) -> None:
        assert write_mzml(small_io_sample.cube, tmp_path / "plain").name == "plain.mzML"


class TestMissingDependency:
    def test_missing_dependency_error_names_the_install_command(self) -> None:
        error = MissingDependencyError("pyopenms", "reading mzML", extra="io")
        message = str(error)
        assert "pyopenms" in message
        assert "pip install 'pyrecycle-analytics[io]'" in message
