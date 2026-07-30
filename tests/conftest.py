"""Shared fixtures.

Full-resolution pyrograms (8700 scans x 372 channels, ~200 components) are what
the platform will run in production, but rebuilding one per test would make the
suite slow enough that nobody runs it. The fixtures here therefore use a reduced
method — fewer scans, a shorter homologous series, a narrower m/z window — while
keeping every structural property the tests care about: the fused PE triplets,
the deliberate co-elutions, the response-factor distortion and the siloxane bleed.

One full-resolution sample is available through :func:`full_resolution_sample` for
the tests that specifically need production dimensions; those are marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_schemas.acquisition import SampleMetadata
from data_schemas.enums import SourceFormat
from data_schemas.pyrogram import PyrogramMetadata
from pyrecycle_analytics.core.binning import axis_to_spec
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.ingestion.mzml import pyopenms_available
from tests.synthetic_data import RECIPES, SyntheticPyrogram, SyntheticPyrogramGenerator


@pytest.fixture(scope="session")
def generator() -> SyntheticPyrogramGenerator:
    """Reduced-method generator used by most tests.

    The retention window matches the reference method exactly, so tabulated marker
    retention times are used unscaled and the intended co-elutions land where the
    reference-spectra module says they do.
    """
    return SyntheticPyrogramGenerator(
        seed=20240517,
        rt_start_s=60.0,
        rt_end_s=1800.0,
        scan_rate_hz=2.0,
        mz_low=29,
        mz_high=320,
        carbon_range=(6, 20),
    )


@pytest.fixture(scope="session")
def hdpe_sample(generator: SyntheticPyrogramGenerator) -> SyntheticPyrogram:
    """Clean HDPE reference: the pure polyolefin comb."""
    return generator.generate(RECIPES["virgin_hdpe"])


@pytest.fixture(scope="session")
def ldpe_sample(generator: SyntheticPyrogramGenerator) -> SyntheticPyrogram:
    """Clean LDPE reference: same comb, four times the branching."""
    return generator.generate(RECIPES["virgin_ldpe"])


@pytest.fixture(scope="session")
def aged_hdpe_sample(generator: SyntheticPyrogramGenerator) -> SyntheticPyrogram:
    """Heavily reprocessed HDPE, for degradation-marker assertions."""
    return generator.generate(RECIPES["aged_hdpe"])


@pytest.fixture(scope="session")
def ps_sample(generator: SyntheticPyrogramGenerator) -> SyntheticPyrogram:
    """Clean PS reference, for marker-triad assertions."""
    return generator.generate(RECIPES["virgin_ps"])


@pytest.fixture(scope="session")
def mixed_sample(generator: SyntheticPyrogramGenerator) -> SyntheticPyrogram:
    """Realistic mixed-polyolefin PCR fraction with percent-level foreign polymers."""
    return generator.generate(RECIPES["pcr_mixed_polyolefin"])


@pytest.fixture(scope="session")
def coelution_sample(generator: SyntheticPyrogramGenerator) -> SyntheticPyrogram:
    """Worst-case overlap blend, for curve-resolution difficulty assertions."""
    return generator.generate(RECIPES["coelution_stress"])


@pytest.fixture(scope="session")
def noiseless_sample(generator: SyntheticPyrogramGenerator) -> SyntheticPyrogram:
    """Exactly bilinear cube: no baseline, no noise, ``D == C @ S``."""
    return generator.generate(RECIPES["coelution_stress"], noiseless=True)


@pytest.fixture(scope="session")
def full_resolution_sample() -> SyntheticPyrogram:
    """A production-dimension pyrogram, for tests that need the real size."""
    return SyntheticPyrogramGenerator(seed=1).generate(RECIPES["pcr_mixed_polyolefin"])


@pytest.fixture
def tiny_cube() -> PyrogramDataCube:
    """A hand-built 6-scan x 5-channel cube with exactly known contents.

    Used where an assertion should read off a number rather than a tolerance.
    """
    retention_times = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    mz_axis = np.array([40.0, 41.0, 42.0, 43.0, 44.0])
    intensities = np.array(
        [
            [0.0, 1.0, 0.0, 2.0, 0.0],
            [1.0, 4.0, 0.0, 6.0, 1.0],
            [2.0, 9.0, 1.0, 12.0, 2.0],
            [1.0, 5.0, 0.0, 7.0, 1.0],
            [0.0, 2.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    metadata = PyrogramMetadata(
        sample=SampleMetadata(sample_id="tiny"),
        source_format=SourceFormat.SYNTHETIC,
        n_scans=retention_times.size,
        mz_axis=axis_to_spec(mz_axis),
        rt_start_s=float(retention_times[0]),
        rt_end_s=float(retention_times[-1]),
    )
    return PyrogramDataCube(
        retention_times=retention_times,
        mz_axis=mz_axis,
        intensities=intensities,
        metadata=metadata,
    )


@pytest.fixture(scope="session")
def small_io_sample() -> SyntheticPyrogram:
    """A deliberately small pyrogram for file round-trip tests.

    Writing an 8700-scan cube to mzML produces a ~40 MB file; this one is ~1 MB and
    exercises exactly the same code paths.
    """
    generator = SyntheticPyrogramGenerator(
        seed=99,
        rt_start_s=60.0,
        rt_end_s=460.0,
        scan_rate_hz=2.0,
        mz_low=29,
        mz_high=200,
        carbon_range=(6, 12),
    )
    return generator.generate(RECIPES["virgin_hdpe"])


requires_pyopenms = pytest.mark.skipif(
    not pyopenms_available(),
    reason="pyopenms is an optional dependency; install with pip install 'pyrecycle-analytics[io]'",
)
"""Skip marker for the tests that need the optional XML parser."""
