"""Rayleigh lidar-constant step: L1 nights -> one CALIBRATION file per day.

This module uses the ``calibration`` package (``eprofile-calibration``) unchanged.
That package writes its own NetCDF under ``options.folder_output``; here that folder
is a throw-away temporary directory, so the only persisted product is the
``CALIBRATION`` file in the :class:`~ceilometer_toolbox.data.CeilometerArchive`.

The night that belongs to a day ``D`` runs from ``D-1`` to ``D``, which matches the
``date`` convention of the calibration package.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr
from calibration import calibrate_rayleigh
from calibration import CalibrationOptions
from calibration import CalibrationResult
from calibration import InstrumentInfo
from calibration.config import DataLevel
from calibration.io.data_loader import load_data

logger = logging.getLogger(__name__)

# Same codes as the calibration package: calibration_method flag_values 0=Rayleigh.
RAYLEIGH_METHOD = 0
# "Exception during calibration" in the calibration flag table (code issue, not physics).
FLAG_EXCEPTION = -99
# "Missing data" fallback when the L1 files cannot be read at all.
FLAG_NO_DATA = 0


def calibrate_night(
        day: date,
        l1_files: list[Path],
        instrument: InstrumentInfo,
        options: CalibrationOptions,
) -> CalibrationResult:
    """Rayleigh calibration of the night that ends on ``day``.

    :param day: the day the calibration is attributed to (night ``day-1`` -> ``day``)
    :param l1_files: the L1 files of ``day-1`` and ``day``, in chronological order
    :param instrument: instrument metadata used by the calibration package
    :param options: calibration options; ``folder_output`` and ``plot_main`` are
        overridden so nothing is written to a second archive
    :return: the calibration result; failures are returned with a flag, never raised
    """
    with tempfile.TemporaryDirectory() as scratch:
        run_options = replace(
            options,
            folder_output=Path(scratch),
            plot_main=False,
        )
        try:
            # The calibration package only reads the L1 files; we load them here so
            # that it never falls back to its own folder_root layout.
            data = load_data(l1_files, instrument.instrument_type, DataLevel.L1)
            if data is None:
                return CalibrationResult(
                    lidar_constant=np.nan,
                    flag=FLAG_NO_DATA,
                    uncertainty=np.nan,
                    message='No readable L1 data',
                )
            return calibrate_rayleigh(
                day.strftime('%Y%m%d'),
                instrument,
                run_options,
                preloaded_data=data,
            )
        except Exception as exc:  # recorded as a flag, like the calibration package
            logger.exception('Rayleigh calibration failed for %s', day)
            return CalibrationResult(
                lidar_constant=np.nan,
                flag=FLAG_EXCEPTION,
                uncertainty=np.nan,
                message=f'{type(exc).__name__}: {exc}',
            )


def _nan_if_none(value: float | None) -> float:
    return np.nan if value is None else float(value)


def write_calibration_file(
        path: str,
        day: date,
        result: CalibrationResult,
        *,
        version: int,
) -> None:
    """Write one day's Rayleigh calibration as a single-row NetCDF file.

    :param path: the archive path prepared by ``atomic_put_file``
    :param day: the day the calibration is attributed to
    :param result: the calibration result
    :param version: molecular-method version code (see ``calibration.io.output``)
    """
    epoch_days = float((day - date(1970, 1, 1)).days)
    ds = xr.Dataset(
        data_vars={
            'lidar_constant': ('time', [float(result.lidar_constant)]),
            'lidar_constant_uncertainty': ('time', [float(result.uncertainty)]),
            'calibration_bottom_height': (
                'time', [_nan_if_none(result.calibration_bottom_height)],
            ),
            'calibration_top_height': (
                'time', [_nan_if_none(result.calibration_top_height)],
            ),
            'flag': ('time', [float(result.flag)]),
            'calibration_method': ('time', [RAYLEIGH_METHOD]),
            'calibration_version': ('time', [version]),
        },
        coords={
            'time': (
                'time', [epoch_days], {
                    'units': 'days since 1970-01-01 00:00:00',
                    'calendar': 'gregorian',
                    'long_name': 'Day the calibration is attributed to (night ends on this day)',
                },
            ),
        },
        attrs={'message': result.message, 'Conventions': 'CF-1.8'},
    )
    ds.to_netcdf(path, engine='netcdf4')
