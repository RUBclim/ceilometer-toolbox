from __future__ import annotations

from datetime import date
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from calibration import CalibrationOptions
from calibration import CalibrationResult
from calibration import InstrumentInfo
from calibration.config import InstrumentType
from ceilometer_toolbox import rayleigh
from ceilometer_toolbox.data import CeilometerArchive
from ceilometer_toolbox.device import Ceilometer


@pytest.fixture
def instrument() -> InstrumentInfo:
    return InstrumentInfo(
        site_name='PAYERNE',
        wmo_id='0-20000-0-06610',
        identifier='A',
        instrument_type=InstrumentType.CL61,
        latitude=46.81,
        longitude=6.94,
        altitude=490.0,
    )


def _result(**kwargs: object) -> CalibrationResult:
    values: dict[str, object] = dict(
        lidar_constant=2.1,
        flag=1,
        uncertainty=0.05,
        calibration_bottom_height=2000.0,
        calibration_top_height=6000.0,
        message='Successful',
    )
    values.update(kwargs)
    return CalibrationResult(**values)


def _first(ds: xr.Dataset, name: str) -> float:
    return float(ds[name].values[0])


def test_write_calibration_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / 'calib.nc'
    rayleigh.write_calibration_file(
        str(path), date(2024, 1, 15), _result(), version=200,
    )
    with xr.open_dataset(path) as ds:
        assert _first(ds, 'lidar_constant') == pytest.approx(2.1)
        assert _first(ds, 'lidar_constant_uncertainty') == pytest.approx(0.05)
        assert _first(ds, 'calibration_bottom_height') == pytest.approx(2000.0)
        assert _first(ds, 'calibration_top_height') == pytest.approx(6000.0)
        assert _first(ds, 'flag') == 1.0
        assert _first(ds, 'calibration_method') == rayleigh.RAYLEIGH_METHOD
        assert _first(ds, 'calibration_version') == 200
        assert ds.attrs['message'] == 'Successful'
        assert ds['time'].values[0] == np.datetime64('2024-01-15')


def test_write_calibration_file_missing_heights_are_nan(tmp_path: Path) -> None:
    path = tmp_path / 'calib.nc'
    result = _result(
        lidar_constant=np.nan,
        flag=-99,
        calibration_bottom_height=None,
        calibration_top_height=None,
        message='RuntimeError: boom',
    )
    rayleigh.write_calibration_file(
        str(path), date(2024, 1, 15), result, version=200,
    )
    with xr.open_dataset(path) as ds:
        assert np.isnan(_first(ds, 'calibration_bottom_height'))
        assert np.isnan(_first(ds, 'calibration_top_height'))
        assert _first(ds, 'flag') == -99.0


def test_calibrate_night_no_readable_data(
        instrument: InstrumentInfo,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rayleigh, 'load_data', lambda *args, **kwargs: None)
    result = rayleigh.calibrate_night(
        date(2024, 1, 15), [], instrument, CalibrationOptions(),
    )
    assert result.flag == rayleigh.FLAG_NO_DATA
    assert result.message == 'No readable L1 data'


def test_calibrate_night_uses_preloaded_data_and_scratch_output(
        instrument: InstrumentInfo,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_calibrate(date_str, info, options, preloaded_data):
        seen['date_str'] = date_str
        seen['preloaded'] = preloaded_data
        seen['folder_output'] = options.folder_output
        seen['plot_main'] = options.plot_main
        return _result()

    monkeypatch.setattr(rayleigh, 'load_data', lambda *a, **k: 'L1-DATA')
    monkeypatch.setattr(rayleigh, 'calibrate_rayleigh', fake_calibrate)

    options = CalibrationOptions(plot_main=True)
    result = rayleigh.calibrate_night(
        date(2024, 1, 15), [Path('a.nc')], instrument, options,
    )

    assert result.flag == 1
    assert seen['date_str'] == '20240115'
    assert seen['preloaded'] == 'L1-DATA'
    assert seen['plot_main'] is False
    assert seen['folder_output'] != options.folder_output
    # the caller's options are not modified
    assert options.plot_main is True


def test_calibrate_night_exception_is_recorded_as_flag(
        instrument: InstrumentInfo,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise ValueError('bad L1')

    monkeypatch.setattr(rayleigh, 'load_data', lambda *a, **k: 'L1-DATA')
    monkeypatch.setattr(rayleigh, 'calibrate_rayleigh', boom)

    result = rayleigh.calibrate_night(
        date(2024, 1, 15), [], instrument, CalibrationOptions(),
    )
    assert result.flag == rayleigh.FLAG_EXCEPTION
    assert np.isnan(result.lidar_constant)
    assert result.message == 'ValueError: bad L1'


def _touch_l1(archive: CeilometerArchive, day: date) -> None:
    path = archive.put_file('IA', 'L1', day, override=True)
    Path(path).touch()


def test_process_rayleigh_calibration_writes_one_file_per_day(
        tmp_path: Path,
        instrument: InstrumentInfo,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = CeilometerArchive(str(tmp_path))
    # L1 exists for day 2 and day 3 only, so day 1 and day 4 are skipped
    _touch_l1(archive, date(2024, 1, 2))
    _touch_l1(archive, date(2024, 1, 3))

    calls: list[tuple[date, list[Path]]] = []

    def fake_night(day, l1_files, info, options):
        calls.append((day, l1_files))
        return _result(lidar_constant=float(day.day))

    monkeypatch.setattr('ceilometer_toolbox.device.calibrate_night', fake_night)
    ceilometer = Ceilometer('IA', input_dir='.', archive=archive)

    options_file = tmp_path / 'options.json'
    options_file.write_text('{}')
    ret = ceilometer.process_rayleigh_calibration(
        options_file=str(options_file),
        start_date='2024-01-01',
        end_date='2024-01-04',
    )

    assert ret == 0
    assert [c[0] for c in calls] == [date(2024, 1, 2), date(2024, 1, 3)]
    # night of day 2 uses the files of day 1 and day 2 (day 1 missing -> only day 2)
    assert calls[0][1] == [
        Path(
            archive.get_file_or_none(
                'IA', 'L1', date(2024, 1, 2),
            ),
        ),
    ]

    out_2 = archive.get_file_or_none('IA', 'CALIBRATION', date(2024, 1, 2))
    out_3 = archive.get_file_or_none('IA', 'CALIBRATION', date(2024, 1, 3))
    assert out_2 is not None and out_3 is not None
    assert archive.get_file_or_none('IA', 'CALIBRATION', date(2024, 1, 1)) is None
    with xr.open_dataset(out_3) as ds:
        assert _first(ds, 'lidar_constant') == pytest.approx(3.0)


def test_process_rayleigh_calibration_defaults_end_to_today(
        tmp_path: Path,
        instrument: InstrumentInfo,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = CeilometerArchive(str(tmp_path))
    today = date.today()
    _touch_l1(archive, today)
    monkeypatch.setattr(
        'ceilometer_toolbox.device.calibrate_night',
        lambda *a, **k: _result(),
    )
    options_file = tmp_path / 'options.json'
    options_file.write_text('{}')
    ceilometer = Ceilometer('IA', input_dir='.', archive=archive)

    ceilometer.process_rayleigh_calibration(
        options_file=str(options_file),
        start_date=today - timedelta(days=1),
    )
    assert archive.get_file_or_none('IA', 'CALIBRATION', today) is not None
