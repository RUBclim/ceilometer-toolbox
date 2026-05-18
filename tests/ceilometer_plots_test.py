import os
from contextlib import contextmanager
from datetime import datetime
from datetime import timedelta
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from ceilometer_toolbox.data import CeilometerArchive
from ceilometer_toolbox.device import Ceilometer

from testing.utils import assert_plot_is_equal

_HERE = os.path.dirname(__file__)
_ROOT = os.path.join(_HERE, '..')
TESTING_OUTPUT = os.path.abspath(os.path.join(_ROOT, 'testing', 'output'))
PLOT_BASELINE = os.path.abspath(
    os.path.join(_ROOT, 'testing', 'plot_baseline'),
)


@pytest.fixture
def ceilometer():
    archive = CeilometerArchive(TESTING_OUTPUT)
    return Ceilometer(
        device_id='IA',
        input_dir='/dev/null',
        archive=archive,
    )


# ---------------------------------------------------------------------------
# beta_plot
# ---------------------------------------------------------------------------

def test_beta_plot_1d(tmp_path, ceilometer):
    fig = ceilometer.beta_plot(
        start_date=datetime(2026, 3, 25, 23, 0),
        end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=1),
        output_path=str(tmp_path / 'L1_beta_1d.jpg'),
        beta_file_type='L1',
    )
    assert_plot_is_equal(
        fig,
        baseline=os.path.join(PLOT_BASELINE, 'L1_beta_1d.jpg'),
    )


def test_beta_plot_1d_alt_max(tmp_path, ceilometer):
    fig = ceilometer.beta_plot(
        start_date=datetime(2026, 3, 25, 23, 0),
        end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=1),
        output_path=str(tmp_path / 'L1_beta_1d_alt_max.jpg'),
        alt_max=4500,
        beta_file_type='L1',
    )
    assert_plot_is_equal(
        fig,
        baseline=os.path.join(PLOT_BASELINE, 'L1_beta_1d_alt_max.jpg'),
    )


def test_beta_plot_30d(tmp_path, ceilometer):
    fig = ceilometer.beta_plot(
        start_date=datetime(2026, 3, 25, 23, 0),
        end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=30),
        output_path=str(tmp_path / 'L1_beta_30d.jpg'),
        beta_file_type='L1',
    )
    assert_plot_is_equal(
        fig,
        baseline=os.path.join(PLOT_BASELINE, 'L1_beta_30d.jpg'),
    )


def test_beta_plot_show_ablh(tmp_path, ceilometer):
    strat_ds = xr.Dataset(
        data_vars={
            'ABLH': (['time'], np.array([500.0, 600.0])),
            'MLH': (['time'], np.array([400.0, 500.0])),
            'quality_FLAG': (['time'], np.array([0, 0])),
            'precip_FLAG': (['time'], np.array([0, 0])),
        },
        coords={
            'time': [
                np.datetime64('2026-03-25T23:30:00'),
                np.datetime64('2026-03-26T00:30:00'),
            ],
        },
    )

    with mock.patch.object(
        ceilometer.archive,
        'open_dataset',
        side_effect=[
            ceilometer.archive.open_dataset(
                'IA', 'L1',
                start_date=datetime(2026, 3, 25, 23, 0),
                end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=1),
                engine='netcdf4',
                data_vars='minimal',
                compat='override',
                coords='minimal',
            ),
            _fake_ctx(strat_ds),
        ],
    ):
        fig = ceilometer.beta_plot(
            start_date=datetime(2026, 3, 25, 23, 0),
            end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=1),
            output_path=str(tmp_path / 'beta_ablh.jpg'),
            show_ablh=True,
        )
    assert fig is not None


def test_beta_plot_show_mlh_and_cbh(tmp_path, ceilometer):
    strat_ds = xr.Dataset(
        data_vars={
            'MLH': (['time'], np.array([400.0, 500.0])),
            'cloud_base_altitude': (
                ['time', 'altitude'],
                np.array([[1000.0, np.nan], [1100.0, np.nan]]),
            ),
            'quality_FLAG': (['time'], np.array([0, 0])),
            'precip_FLAG': (['time'], np.array([0, 0])),
        },
        coords={
            'time': [
                np.datetime64('2026-03-25T23:30:00'),
                np.datetime64('2026-03-26T00:30:00'),
            ],
            'altitude': [1000.0, 2000.0],
        },
    )

    with mock.patch.object(
        ceilometer.archive,
        'open_dataset',
        side_effect=[
            ceilometer.archive.open_dataset(
                'IA', 'L1',
                start_date=datetime(2026, 3, 25, 23, 0),
                end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=1),
                engine='netcdf4',
                data_vars='minimal',
                compat='override',
                coords='minimal',
            ),
            _fake_ctx(strat_ds),
        ],
    ):
        fig = ceilometer.beta_plot(
            start_date=datetime(2026, 3, 25, 23, 0),
            end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=1),
            output_path=str(tmp_path / 'beta_mlh_cbh.jpg'),
            show_mlh=True,
            show_cbh=True,
        )
    assert fig is not None


def test_beta_plot_filter_qc_false(tmp_path, ceilometer):
    strat_ds = xr.Dataset(
        data_vars={
            'ABLH': (['time'], np.array([500.0])),
            'quality_FLAG': (['time'], np.array([1])),
            'precip_FLAG': (['time'], np.array([0])),
        },
        coords={
            'time': [np.datetime64('2026-03-26T00:00:00')],
        },
    )

    with mock.patch.object(
        ceilometer.archive,
        'open_dataset',
        side_effect=[
            ceilometer.archive.open_dataset(
                'IA', 'L1',
                start_date=datetime(2026, 3, 25, 23, 0),
                end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=1),
                engine='netcdf4',
                data_vars='minimal',
                compat='override',
                coords='minimal',
            ),
            _fake_ctx(strat_ds),
        ],
    ):
        fig = ceilometer.beta_plot(
            start_date=datetime(2026, 3, 25, 23, 0),
            end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=1),
            output_path=str(tmp_path / 'beta_no_qc.jpg'),
            show_ablh=True,
            filter_qc=False,
        )
    assert fig is not None


# ---------------------------------------------------------------------------
# beta_plot — L2A_beta coordinate fallback
# ---------------------------------------------------------------------------

def test_beta_plot_l2a_beta_coord_fallback(tmp_path, ceilometer):
    # Exercises the except NotImplementedError branch: when station_latitude/
    # longitude do not support .item() (e.g. time-varying coords in L2A_beta
    # files), beta_plot falls back to .values[0].
    times = [
        np.datetime64('2026-03-25T23:30:00'),
        np.datetime64('2026-03-26T00:30:00'),
    ]
    ds = xr.Dataset(
        data_vars={
            'beta': (['time', 'range'], np.full((2, 2), 1e-6)),
            'station_latitude': (['time'], np.array([51.0, 51.0])),
            'station_longitude': (['time'], np.array([7.0, 7.0])),
        },
        coords={'time': times, 'range': [100.0, 200.0]},
    )

    orig_item = xr.DataArray.item

    def _raise_for_coord(self):
        if self.name in ('station_latitude', 'station_longitude'):
            raise NotImplementedError
        return orig_item(self)

    with mock.patch.object(
        ceilometer.archive,
        'open_dataset',
        return_value=_fake_ctx(ds),
    ):
        with mock.patch.object(xr.DataArray, 'item', _raise_for_coord):
            fig = ceilometer.beta_plot(
                start_date=datetime(2026, 3, 25, 23, 0),
                end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=1),
                output_path=str(tmp_path / 'beta_l2a_fallback.jpg'),
            )
    assert fig is not None


# ---------------------------------------------------------------------------
# ldr_plot
# ---------------------------------------------------------------------------

def test_ldr_plot_1d(tmp_path, ceilometer):
    fig = ceilometer.ldr_plot(
        start_date=datetime(2026, 3, 25, 23, 0),
        end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=1),
        output_path=str(tmp_path / 'L1_ldr_1d.jpg'),
    )
    assert_plot_is_equal(
        fig,
        baseline=os.path.join(PLOT_BASELINE, 'L1_ldr_1d.jpg'),
    )


def test_ldr_plot_1d_alt_max(tmp_path, ceilometer):
    fig = ceilometer.ldr_plot(
        start_date=datetime(2026, 3, 25, 23, 0),
        end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=1),
        output_path=str(tmp_path / 'L1_ldr_1d_alt_max.jpg'),
        alt_max=4500,
    )
    assert_plot_is_equal(
        fig,
        baseline=os.path.join(PLOT_BASELINE, 'L1_ldr_1d_alt_max.jpg'),
    )


def test_ldr_plot_30d(tmp_path, ceilometer):
    fig = ceilometer.ldr_plot(
        start_date=datetime(2026, 3, 25, 23, 0),
        end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=30),
        output_path=str(tmp_path / 'L1_ldr_30d.jpg'),
    )
    assert_plot_is_equal(
        fig,
        baseline=os.path.join(PLOT_BASELINE, 'L1_ldr_30d.jpg'),
    )


def test_ldr_plot_raises_when_linear_depol_ratio_missing(tmp_path, ceilometer):
    # Exercises the KeyError branch added to ldr_plot when the dataset does
    # not contain the linear_depol_ratio variable.
    ds = xr.Dataset(
        data_vars={
            'station_latitude': xr.Variable([], 51.0),
            'station_longitude': xr.Variable([], 7.0),
        },
        coords={'time': [np.datetime64('2026-03-26T00:00:00')]},
    )

    with mock.patch.object(
        ceilometer.archive,
        'open_dataset',
        return_value=_fake_ctx(ds),
    ):
        with pytest.raises(
            KeyError,
            match='linear_depol_ratio variable is not available',
        ):
            ceilometer.ldr_plot(
                start_date=datetime(2026, 3, 25, 23, 0),
                end_date=datetime(2026, 3, 25, 23, 0) + timedelta(days=1),
                output_path=str(tmp_path / 'ldr_err.jpg'),
            )


# ---------------------------------------------------------------------------
# Helper: fake context manager wrapping a dataset
# ---------------------------------------------------------------------------


@contextmanager
def _fake_ctx(ds):
    yield ds
