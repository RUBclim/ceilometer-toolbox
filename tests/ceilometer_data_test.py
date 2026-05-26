import os
import stat
from datetime import date
from datetime import datetime
from unittest import mock

import pytest
import xarray as xr
from ceilometer_toolbox.data import atomic_write_path
from ceilometer_toolbox.data import CeilometerArchive


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        file.write('test')


def test_init_rejects_non_existing_root(tmp_path):
    missing = os.path.join(tmp_path, 'missing-root')
    with pytest.raises(ValueError, match='Root directory does not exist'):
        CeilometerArchive(missing)


def test_parse_date_accepts_supported_inputs(tmp_path):
    archive = CeilometerArchive(tmp_path)
    assert archive._parse_date(date(2026, 3, 1)) == date(2026, 3, 1)
    assert archive._parse_date(
        datetime(2026, 3, 2, 5, 0, 0),
    ) == date(2026, 3, 2)
    assert archive._parse_date('2026-03-03') == date(2026, 3, 3)
    assert archive._parse_date('20260304') == date(2026, 3, 4)


def test_parse_date_rejects_invalid_value(tmp_path):
    archive = CeilometerArchive(tmp_path)
    with pytest.raises(ValueError, match='Date values must be'):
        archive._parse_date('03-04-2026')


def test_parse_date_rejects_unsupported_type(tmp_path):
    archive = CeilometerArchive(tmp_path)
    with pytest.raises(ValueError, match='Date values must be'):
        archive._parse_date(123)  # type: ignore[arg-type]


def test_iter_days_is_inclusive(tmp_path):
    archive = CeilometerArchive(tmp_path)
    days = list(archive._iter_days(date(2026, 1, 1), date(2026, 1, 3)))
    assert days == [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]


def test_iter_device_ids_for_explicit_device_existing_and_missing(tmp_path):
    os.makedirs(os.path.join(tmp_path, 'IA'))
    archive = CeilometerArchive(tmp_path)
    assert list(archive._iter_device_ids('IA')) == ['IA']
    assert list(archive._iter_device_ids('IX')) == []


def test_iter_device_ids_lists_only_directories(tmp_path):
    os.makedirs(os.path.join(tmp_path, 'IB'))
    _touch(os.path.join(tmp_path, 'not-a-device.txt'))
    archive = CeilometerArchive(tmp_path)
    assert list(archive._iter_device_ids()) == ['IB']


def test_iter_device_ids_returns_empty_when_root_unavailable(tmp_path):
    archive = CeilometerArchive(tmp_path)
    os.rmdir(tmp_path)

    assert list(archive._iter_device_ids()) == []


def test_available_files_returns_empty_for_invalid_range(tmp_path):
    archive = CeilometerArchive(tmp_path)

    assert archive._available_files(
        device_id='IA',
        file_type='L1',
        start_date=date(2026, 3, 2),
        end_date=date(2026, 3, 1),
    ) == []
    assert archive._available_files(device_id='IA') == []


def test_available_files_partial_range_filters_names_and_bounds(tmp_path):
    paths = [
        'IA/2026/02/20260220_L1.nc',
        'IA/2026/03/20260301_L1.nc',
        'IA/2026/03/20260302_L2A_beta.nc',
        'IA/2026/03/20260303_L1.nc',
        'IA/2026/03/README.txt',
    ]
    for path in paths:
        _touch(os.path.join(tmp_path, path))

    archive = CeilometerArchive(tmp_path)

    rows = archive._available_files(
        device_id='IA',
        file_type='L1',
        start_date=date(2026, 3, 1),
        end_date=None,
    )

    assert [os.path.basename(row.full_path) for row in rows] == [
        '20260301_L1.nc',
        '20260303_L1.nc',
    ]


def test_available_files_filters_with_start_only(tmp_path):
    paths = [
        'IA/2025/12/20251231_L1.nc',
        'IA/2026/01/20260101_L1.nc',
        'IA/2026/02/20260201_L1.nc',
        'IA/2026/02/20260215_L1.nc',
    ]
    for path in paths:
        _touch(os.path.join(tmp_path, path))

    archive = CeilometerArchive(tmp_path)

    rows = archive._available_files(
        device_id='IA',
        file_type='L1',
        start_date=date(2026, 2, 10),
        end_date=None,
    )

    assert [
        os.path.basename(row.full_path)
        for row in rows
    ] == ['20260215_L1.nc']


def test_available_files_filters_with_end_only(tmp_path):
    paths = [
        'IA/2026/02/20260215_L1.nc',
        'IA/2026/02/20260225_L1.nc',
        'IA/2026/03/20260315_L1.nc',
        'IA/2027/01/20270101_L1.nc',
    ]
    for path in paths:
        _touch(os.path.join(tmp_path, path))

    archive = CeilometerArchive(tmp_path)

    rows = archive._available_files(
        device_id='IA',
        file_type='L1',
        start_date=None,
        end_date=date(2026, 2, 20),
    )

    assert [
        os.path.basename(row.full_path)
        for row in rows
    ] == ['20260215_L1.nc']


def test_validate_file_type_accepts_and_rejects(tmp_path):
    archive = CeilometerArchive(tmp_path)

    archive._validate_file_type('L1')
    with pytest.raises(ValueError, match='Unsupported file type'):
        archive._validate_file_type('bad')


def test_file_path_builds_and_rejects_invalid_type(tmp_path):
    archive = CeilometerArchive(tmp_path)

    out = archive._file_path('IA', 'L2A_stratfinder', '2026-04-01')
    assert out.endswith('/IA/2026/04/20260401_L2A_stratfinder.nc')

    with pytest.raises(ValueError, match='Unsupported file type'):
        archive._file_path('IA', 'bad', '2026-04-01')  # type: ignore[arg-type]


def test_list_files_inclusive_date_range(tmp_path):
    paths = [
        'IA/2026/02/20260217_L1.nc',
        'IA/2026/02/20260218_L1.nc',
        'IA/2026/02/20260219_L1.nc',
        'IA/2026/02/20260220_L1.nc',
    ]
    for path in paths:
        _touch(os.path.join(tmp_path, path))

    archive = CeilometerArchive(tmp_path)

    files = archive.get_files('IA', 'L1', date(2026, 2, 17), date(2026, 2, 20))

    assert [os.path.basename(p) for p in files] == [
        '20260217_L1.nc',
        '20260218_L1.nc',
        '20260219_L1.nc',
        '20260220_L1.nc',
    ]


def test_iter_files_filters_missing_days(tmp_path):
    for path in [
        'IA/2026/02/20260221_L2A_stratfinder.nc',
        'IA/2026/02/20260223_L2A_stratfinder.nc',
    ]:
        _touch(os.path.join(tmp_path, path))

    archive = CeilometerArchive(tmp_path)

    files = list(
        archive.iter_files(
            'IA', 'L2A_stratfinder', '2026-02-21', '2026-02-23',
        ),
    )

    assert [os.path.basename(p) for p in files] == [
        '20260221_L2A_stratfinder.nc', '20260223_L2A_stratfinder.nc',
    ]


def test_iter_files_invalid_file_type(tmp_path):
    archive = CeilometerArchive(tmp_path)

    with pytest.raises(ValueError, match='Unsupported file type'):
        list(
            archive.iter_files(
                'IA',
                'foo',  # type: ignore[arg-type]
                '2026-03-01',
                '2026-03-02',
            ),
        )


def test_iter_files_invalid_date_order(tmp_path):
    archive = CeilometerArchive(tmp_path)

    with pytest.raises(ValueError, match='start_date must be <= end_date'):
        list(archive.iter_files('IA', 'L1', '2026-03-03', '2026-03-02'))


def test_list_files_defaults_all_devices_types_and_date_range(tmp_path):
    for path in [
        'IA/2026/02/20260217_L1.nc',
        'IA/2026/02/20260218_L2A_beta.nc',
        'IB/2026/03/20260301_L2A_stratfinder.nc',
    ]:
        _touch(os.path.join(tmp_path, path))

    archive = CeilometerArchive(tmp_path)

    files = archive.get_files()

    assert sorted(os.path.basename(p) for p in files) == [
        '20260217_L1.nc',
        '20260218_L2A_beta.nc',
        '20260301_L2A_stratfinder.nc',
    ]


def test_get_files_device_filter_isolates_per_station(tmp_path):
    for path in [
        'IA/2026/03/20260301_L1.nc',
        'IB/2026/03/20260301_L1.nc',
        'IB/2026/03/20260302_L1.nc',
    ]:
        _touch(os.path.join(tmp_path, path))

    archive = CeilometerArchive(tmp_path)

    ia_files = archive.get_files(device_id='IA', file_type='L1')
    ib_files = archive.get_files(device_id='IB', file_type='L1')

    assert [os.path.basename(p) for p in ia_files] == ['20260301_L1.nc']
    assert [os.path.basename(p) for p in ib_files] == [
        '20260301_L1.nc', '20260302_L1.nc',
    ]


def test_put_and_delete_file_multiple_devices(tmp_path):
    archive = CeilometerArchive(tmp_path)

    for device_id in ('IA', 'IB'):
        path = archive.put_file(device_id, 'L1', '2026-03-01')
        _touch(path)
        assert archive.get_file_or_none(device_id, 'L1', '2026-03-01') == path

    assert archive.delete_file('IA', 'L1', '2026-03-01') is True
    assert archive.get_file_or_none('IA', 'L1', '2026-03-01') is None
    # IB file is unaffected
    assert archive.get_file_or_none('IB', 'L1', '2026-03-01') is not None


def test_latest_date_for_device_and_type(tmp_path):
    paths = [
        'IA/2026/03/20260301_L1.nc',
        'IA/2026/03/20260305_L1.nc',
        'IA/2026/03/20260304_L2A_beta.nc',
        'IB/2026/03/20260302_L2A_beta.nc',
    ]
    for path in paths:
        _touch(os.path.join(tmp_path, path))

    archive = CeilometerArchive(tmp_path)

    assert archive.latest_date('IA', 'L1') == date(2026, 3, 5)
    assert archive.latest_date('IA', 'L2A_beta') == date(2026, 3, 4)
    assert archive.latest_date('IB', 'L2A_beta') == date(2026, 3, 2)
    assert archive.latest_date('IC', 'L2A_beta') is None


def test_latest_date_uses_from_date_and_max_depth_days(tmp_path):
    _touch(os.path.join(tmp_path, 'IA/2026/03/20260305_L1.nc'))

    archive = CeilometerArchive(tmp_path)

    assert archive.latest_date(
        'IA', 'L1', from_date='2026-03-10', max_depth_days=4,
    ) is None
    assert archive.latest_date(
        'IA', 'L1', from_date='2026-03-10', max_depth_days=5,
    ) == date(2026, 3, 5)


def test_latest_date_rejects_negative_max_depth_days(tmp_path):
    archive = CeilometerArchive(tmp_path)

    with pytest.raises(ValueError, match='max_depth_days must be >= 0'):
        archive.latest_date('IA', 'L1', max_depth_days=-1)


def test_latest_date_zero_max_depth_days_only_checks_anchor(tmp_path):
    for path in ['IA/2026/03/20260310_L1.nc', 'IA/2026/03/20260309_L1.nc']:
        _touch(os.path.join(tmp_path, path))
    archive = CeilometerArchive(tmp_path)

    assert archive.latest_date(
        'IA', 'L1', from_date='2026-03-10', max_depth_days=0,
    ) == date(2026, 3, 10)
    assert archive.latest_date(
        'IA', 'L1', from_date='2026-03-11', max_depth_days=0,
    ) is None


def test_write_file_creates_tree(tmp_path):
    archive = CeilometerArchive(tmp_path)

    out = archive.put_file(
        device_id='IA',
        file_type='L2A_beta',
        file_date='2026-03-19',
    )
    assert os.path.isdir(os.path.dirname(out))
    assert os.path.exists(out) is False


def test_write_file_respects_override(tmp_path):
    archive = CeilometerArchive(tmp_path)

    out = archive.put_file(
        device_id='IA',
        file_type='L2A_beta',
        file_date='2026-03-19',
    )
    _touch(out)
    assert os.path.isfile(out)

    with pytest.raises(FileExistsError):
        archive.put_file(
            device_id='IA',
            file_type='L2A_beta',
            file_date='2026-03-19',
            override=False,
        )

    archive.put_file(
        device_id='IA',
        file_type='L2A_beta',
        file_date='2026-03-19',
        override=True,
    )
    assert os.path.isfile(out)


def test_delete_file_removes_existing_and_returns_status(tmp_path):
    archive = CeilometerArchive(tmp_path)

    out = archive.put_file(
        device_id='IA',
        file_type='L1',
        file_date='2026-03-20',
    )
    _touch(out)
    assert os.path.isfile(out)

    assert archive.delete_file('IA', 'L1', '2026-03-20') is True
    assert os.path.exists(out) is False
    assert archive.delete_file('IA', 'L1', '2026-03-20') is False


def test_get_file_or_none_returns_none_if_missing(tmp_path):
    archive = CeilometerArchive(tmp_path)

    missing = archive.get_file_or_none('IA', 'L1', '2026-03-21')
    assert missing is None


def test_get_file_or_none_returns_path_if_present(tmp_path):
    archive = CeilometerArchive(tmp_path)

    out = archive.put_file('IA', 'L1', '2026-03-21')
    _touch(out)
    assert archive.get_file_or_none('IA', 'L1', '2026-03-21') == out


def test_put_file_rejects_invalid_type(tmp_path):
    archive = CeilometerArchive(tmp_path)

    with pytest.raises(ValueError, match='Unsupported file type'):
        archive.put_file('IA', 'foo', '2026-03-22')  # type: ignore[arg-type]


def test_atomic_write_path_publishes_file_on_success(tmp_path):
    final_path = os.path.join(tmp_path, 'IA', '2026', '03', '20260322_L1.nc')

    with atomic_write_path(final_path=final_path) as staged:
        with open(staged, 'w', encoding='utf-8') as file:
            file.write('ok')

    with open(final_path, encoding='utf-8') as file:
        assert file.read() == 'ok'

    file_stats = os.stat(final_path)
    # the default umask of NamedTemporaryFile is 0o600, but we want 0o644
    assert oct(stat.S_IMODE(file_stats.st_mode)) == '0o644'


def test_atomic_write_path_removes_staged_file_when_exception(tmp_path):
    final_path = os.path.join(tmp_path, 'IA', '2026', '03', '20260322_L1.nc')
    staged = None

    with pytest.raises(RuntimeError, match='boom'):
        with atomic_write_path(final_path=final_path) as tmp_out:
            staged = tmp_out
            with open(tmp_out, 'w', encoding='utf-8') as file:
                file.write('broken')
            raise RuntimeError('boom')

    assert os.path.exists(final_path) is False  # type: ignore[unreachable]
    assert staged is not None
    assert os.path.exists(staged) is False


def test_atomic_write_path_respects_override_false(tmp_path):
    final_path = os.path.join(tmp_path, 'IA', '2026', '03', '20260322_L1.nc')
    _touch(final_path)

    with pytest.raises(FileExistsError, match='File already exists'):
        with atomic_write_path(final_path=final_path, override=False):
            pass  # pragma: no cover


def test_atomic_write_path_detects_race_on_publish(tmp_path):
    final_path = os.path.join(tmp_path, 'IA', '2026', '03', '20260322_L1.nc')
    staged = None

    with pytest.raises(FileExistsError, match='File already exists'):
        with atomic_write_path(final_path, override=False) as tmp_out:
            staged = tmp_out
            with open(tmp_out, 'w', encoding='utf-8') as file:
                file.write('staged')
            _touch(final_path)

    assert staged is not None
    assert os.path.exists(staged) is False


def test_atomic_put_file_publishes_file_on_success(tmp_path):
    archive = CeilometerArchive(tmp_path)

    with archive.atomic_put_file('IA', 'L1', '2026-03-22') as staged:
        with open(staged, 'w', encoding='utf-8') as file:
            file.write('ok')

    out = archive.get_file_or_none('IA', 'L1', '2026-03-22')
    assert out is not None
    with open(out, encoding='utf-8') as file:
        assert file.read() == 'ok'


def test_atomic_write_path_does_not_publish_when_staged_file_is_deleted(tmp_path):
    final_path = os.path.join(tmp_path, 'IA', '2026', '03', '20260323_L1.nc')

    with atomic_write_path(final_path) as staged:
        os.unlink(staged)

    assert os.path.exists(final_path) is False


def test_available_files_skips_disappeared_device_dir(tmp_path):
    archive = CeilometerArchive(tmp_path)

    with mock.patch.object(
        archive, '_iter_device_ids', return_value=['NONEXISTENT'],
    ):
        rows = archive._available_files(start_date=date(2026, 3, 1))

    assert rows == []


def test_open_dataset_raises_if_no_files(tmp_path):
    archive = CeilometerArchive(tmp_path)

    with pytest.raises(FileNotFoundError, match='No files found for'):
        with archive.open_dataset(device_id='IA', file_type='L1') as _:
            pass  # pragma: no cover


def test_open_dataset_non_beta_passes_kwargs_and_slices(tmp_path):
    archive = CeilometerArchive(tmp_path)
    dataset = xr.Dataset(coords={'time': ['2026-03-01', '2026-03-02']})
    files = []
    for day in ('2026-03-01', '2026-03-02'):
        out = archive.put_file('IA', 'L1', day)
        _touch(out)
        files.append(out)

    with mock.patch(
        'ceilometer_toolbox.data.xr.open_mfdataset',
        return_value=dataset,
    ) as open_mfdataset:
        with archive.open_dataset(
            device_id='IA',
            file_type='L1',
            start_date='2026-03-01',
            end_date='2026-03-02',
            chunks={'time': 1},
        ) as out:

            assert list(out.time.values) == list(dataset.time.values)
            open_mfdataset.assert_called_once_with(files, chunks={'time': 1})
            assert len(out.time.values) == 2


def test_open_dataset_non_beta_without_dates_skips_slice(tmp_path):
    archive = CeilometerArchive(tmp_path)
    dataset = xr.Dataset(coords={'time': ['2026-03-01']})
    out_path = archive.put_file('IA', 'L2A_stratfinder', '2026-03-01')
    _touch(out_path)

    with mock.patch(
        'ceilometer_toolbox.data.xr.open_mfdataset',
        return_value=dataset,
    ):
        with archive.open_dataset('IA', file_type='L2A_stratfinder') as out:
            assert out is dataset
            assert len(out.time.values) == 1


def test_open_dataset_non_beta_with_only_end_date_slices(tmp_path):
    archive = CeilometerArchive(tmp_path)
    dataset = xr.Dataset(coords={'time': ['2026-03-01']})
    out_path = archive.put_file('IA', 'L1', '2026-03-01')
    _touch(out_path)

    with mock.patch(
        'ceilometer_toolbox.data.xr.open_mfdataset',
        return_value=dataset,
    ):
        with archive.open_dataset(
            device_id='IA', file_type='L1', end_date='2026-03-02',
        ) as out:
            assert out.equals(dataset)
            assert len(out.time.values) == 1


def test_open_dataset_beta_uses_preprocess_with_tolerance(tmp_path):
    archive = CeilometerArchive(tmp_path)
    ref_dataset = xr.Dataset(coords={'altitude': [10.0, 20.0, 30.0]})
    files = []
    for day in ('2026-03-01', '2026-03-02'):
        out = archive.put_file('IA', 'L2A_beta', day)
        _touch(out)
        files.append(out)

    seen = {}

    def _fake_open_mfdataset(*args, **kwargs):
        preprocess = kwargs['preprocess']
        raw = xr.Dataset(
            coords={
                'time': ['2026-03-01'],
                'altitude': [9.1, 20.9, 30.8],
            },
        )
        processed = preprocess(raw)
        seen['altitude'] = list(processed.altitude.values)
        return xr.Dataset(coords={'time': ['2026-03-01']})

    with mock.patch(
        'ceilometer_toolbox.data.xr.open_dataset',
        return_value=ref_dataset,
    ) as open_dataset, mock.patch(
        'ceilometer_toolbox.data.xr.open_mfdataset',
        side_effect=_fake_open_mfdataset,
    ) as open_mfdataset:
        with archive.open_dataset(
            device_id='IA', file_type='L2A_beta', combine='nested',
        ) as out:
            assert len(out.time.values) == 1
    open_dataset.assert_called_once_with(files[0])
    open_mfdataset.assert_called_once_with(
        files,
        join='outer',
        preprocess=mock.ANY,
        combine='nested',
    )
    assert seen['altitude'] == [10.0, 20.0, 30.0]
