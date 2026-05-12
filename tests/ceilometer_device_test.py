import os
import shutil
import tempfile
from datetime import date
from unittest import mock

import pytest
import xarray as xr
from ceilometer_toolbox.data import CeilometerArchive
from ceilometer_toolbox.device import Ceilometer
from freezegun import freeze_time

_HERE = os.path.dirname(__file__)
_ROOT = os.path.join(_HERE, '..')
TESTING_INPUT = os.path.join(_ROOT, 'testing', 'input')
TESTING_OUTPUT = os.path.join(_ROOT, 'testing', 'output')
RAW2L1_CONF = os.path.abspath(
    os.path.join(
        _ROOT, 'example_configs', 'raw2l1_cl61.conf',
    ),
)

INPUT_FIXTURES = [
    'live_20260325_235741.nc',
    'live_20260326_001241.nc',
    'live_20260326_002741.nc',
]


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _copy_input_fixtures(input_dir):
    for filename in INPUT_FIXTURES:
        _copy(
            os.path.join(TESTING_INPUT, filename),
            os.path.join(input_dir, filename),
        )


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('test')


def _assert_netcdf_equal(actual_path, expected_path):
    with xr.open_dataset(actual_path) as actual, xr.open_dataset(
            expected_path,
    ) as expected:
        xr.testing.assert_equal(actual, expected)


@pytest.fixture
def dirs(tmp_path):
    input_dir = os.path.join(str(tmp_path), 'input')
    output_dir = os.path.join(str(tmp_path), 'output')
    os.makedirs(input_dir)
    os.makedirs(output_dir)
    return {'input_dir': input_dir, 'output_dir': output_dir}


@pytest.fixture
def archive(dirs):
    return CeilometerArchive(dirs['output_dir'])


@pytest.fixture
def ceilometer(dirs, archive):
    return Ceilometer(
        device_id='IA',
        input_dir=dirs['input_dir'],
        archive=archive,
        raw2l1_config_file=RAW2L1_CONF,
    )


# ---------------------------------------------------------------------------
# glob_day_raw_data
# ---------------------------------------------------------------------------

def test_glob_day_raw_data_returns_only_matching_files(dirs, ceilometer):
    _copy_input_fixtures(dirs['input_dir'])
    _copy(
        os.path.join(TESTING_INPUT, 'live_20260325_235741.nc'),
        os.path.join(dirs['input_dir'], 'other_20260326_130000.nc'),
    )

    files = sorted(
        ceilometer.glob_day_raw_data(
            date(2026, 3, 26), prefix='live_',
        ),
    )

    assert [os.path.basename(p) for p in files] == [
        'live_20260326_001241.nc',
        'live_20260326_002741.nc',
    ]
    assert ceilometer.glob_day_raw_data(
        date(2026, 3, 24), prefix='live_',
    ) == []


# ---------------------------------------------------------------------------
# to_l1
# ---------------------------------------------------------------------------

def test_to_l1_requires_config_file(dirs, archive):
    cel = Ceilometer(
        device_id='IA', input_dir=dirs['input_dir'], archive=archive,
    )
    with pytest.raises(ValueError, match='config_file must be provided'):
        cel.to_l1(
            file_date=date(2026, 3, 25),
            input_files=[],
            output_file='/tmp/out.nc',
        )


def test_to_l1_runs_and_matches_fixture_output(dirs, ceilometer):
    _copy_input_fixtures(dirs['input_dir'])
    output_file = os.path.join(dirs['output_dir'], '20260325_L1.nc')
    log_file = os.path.join(dirs['output_dir'], 'raw2l1.log')

    ret = ceilometer.to_l1(
        file_date=date(2026, 3, 25),
        input_files=[
            os.path.join(dirs['input_dir'], 'live_20260325_235741.nc'),
            os.path.join(dirs['input_dir'], 'live_20260326_001241.nc'),
        ],
        output_file=output_file,
        ancillary_files=os.path.join(
            dirs['input_dir'], 'live_20260326_001241.nc',
        ),
        min_file_size=0,
        filter_max_age=50000,
        filter_day=True,
        log_file=log_file,
        log_level='INFO',
        verbose='INFO',
    )

    assert ret == 0
    assert os.path.exists(output_file)
    assert os.path.exists(log_file)
    _assert_netcdf_equal(
        output_file,
        os.path.join(TESTING_OUTPUT, 'IA', '2026', '03', '20260325_L1.nc'),
    )


def test_to_l1_with_explicit_config_file(dirs, archive):
    # Exercises the False branch of `if not config_file:` in to_l1
    cel = Ceilometer(
        device_id='IA', input_dir=dirs['input_dir'], archive=archive,
    )
    _copy(
        os.path.join(TESTING_INPUT, 'live_20260326_001241.nc'),
        os.path.join(dirs['input_dir'], 'live_20260326_001241.nc'),
    )
    output_file = os.path.join(dirs['output_dir'], 'out.nc')

    ret = cel.to_l1(
        file_date=date(2026, 3, 26),
        input_files=os.path.join(dirs['input_dir'], 'live_20260326_001241.nc'),
        output_file=output_file,
        config_file=RAW2L1_CONF,
        filter_day=True,
    )

    assert ret == 0


def test_to_l1_accepts_string_input_and_default_log(dirs, ceilometer):
    _copy(
        os.path.join(TESTING_INPUT, 'live_20260326_001241.nc'),
        os.path.join(dirs['input_dir'], 'live_20260326_001241.nc'),
    )
    output_file = os.path.join(dirs['output_dir'], 'single_file_L1.nc')

    ret = ceilometer.to_l1(
        file_date=date(2026, 3, 26),
        input_files=os.path.join(dirs['input_dir'], 'live_20260326_001241.nc'),
        output_file=output_file,
        filter_day=True,
    )

    assert ret == 0
    assert os.path.exists(output_file)
    assert os.path.exists(
        os.path.join(tempfile.gettempdir(), 'raw2l1_20260326.log'),
    )


def test_to_l1_raises_for_failing_raw2l1(dirs, ceilometer):
    _copy_input_fixtures(dirs['input_dir'])
    output_file = os.path.join(dirs['output_dir'], 'out.nc')

    with pytest.raises(RuntimeError, match='raw2l1 failed with return code'):
        ceilometer.to_l1(
            file_date=date(2026, 3, 25),
            input_files=[
                os.path.join(dirs['input_dir'], 'live_20260325_235741.nc'),
                os.path.join(dirs['input_dir'], 'live_20260326_001241.nc'),
            ],
            output_file=output_file,
            check_timeliness=True,
            filter_day=True,
        )

    assert os.path.exists(output_file) is False


# ---------------------------------------------------------------------------
# process_raw_files
# ---------------------------------------------------------------------------

def test_process_raw_files_with_explicit_config_file(dirs, ceilometer):
    # Exercises the False branch of `if not config_file:` in process_raw_files
    with freeze_time('2026-03-25 12:00:00'):
        ret = ceilometer.process_raw_files(
            start_date=date(2026, 3, 26),
            config_file=RAW2L1_CONF,
        )
    assert ret == 0


def test_process_raw_files_requires_config_file(dirs, archive):
    cel = Ceilometer(
        device_id='IA', input_dir=dirs['input_dir'], archive=archive,
    )
    with pytest.raises(ValueError, match='config_file must be provided'):
        cel.process_raw_files(start_date=date(2026, 3, 25))


def test_process_raw_files_rejects_invalid_date_range(dirs, ceilometer):
    with pytest.raises(ValueError, match='start_date cannot be after end_date'):
        ceilometer.process_raw_files(
            start_date=date(2026, 3, 27),
            end_date=date(2026, 3, 25),
        )


def test_process_raw_files_future_date_short_circuits(dirs, ceilometer):
    _copy_input_fixtures(dirs['input_dir'])

    with freeze_time('2026-03-26 12:00:00'):
        ret = ceilometer.process_raw_files(start_date=date(2026, 3, 27))

    assert ret == 0
    assert os.listdir(dirs['output_dir']) == []


def test_process_raw_files_with_explicit_end_date(dirs, ceilometer):
    _copy_input_fixtures(dirs['input_dir'])

    ret = ceilometer.process_raw_files(
        start_date=date(2026, 3, 25),
        end_date=date(2026, 3, 25),
    )

    assert ret == 0
    assert os.path.exists(
        os.path.join(dirs['output_dir'], 'IA', '2026', '03', '20260325_L1.nc'),
    )


def test_process_raw_files_with_str_end_date(dirs, ceilometer):
    _copy_input_fixtures(dirs['input_dir'])

    ret = ceilometer.process_raw_files(
        start_date=date(2026, 3, 25),
        end_date='2026-03-25',
    )

    assert ret == 0


def test_process_raw_files_with_no_current_day_files(dirs, ceilometer):
    with freeze_time('2026-03-25 12:00:00'):
        ret = ceilometer.process_raw_files(start_date=date(2026, 3, 25))

    assert ret == 0
    assert os.listdir(dirs['output_dir']) == []


def test_process_raw_files_uses_latest_date_when_none(dirs, ceilometer):
    # No L1 files in archive → latest_date returns None → falls back to 1970-01-01
    with freeze_time('1970-01-01 12:00:00'):
        ret = ceilometer.process_raw_files(start_date=None)

    assert ret == 0


def test_process_raw_files_uses_str_start_date(dirs, ceilometer):
    with freeze_time('2026-03-25 12:00:00'):
        ret = ceilometer.process_raw_files(start_date='2026-03-26')

    assert ret == 0


def test_process_raw_files_builds_expected_outputs(dirs, ceilometer):
    _copy_input_fixtures(dirs['input_dir'])

    ret = ceilometer.process_raw_files(
        start_date='2026-03-25',
        end_date=date(2026, 3, 26),
        prefix='live_',
    )

    assert ret == 0
    output_25 = os.path.join(
        dirs['output_dir'], 'IA', '2026', '03', '20260325_L1.nc',
    )
    output_26 = os.path.join(
        dirs['output_dir'], 'IA', '2026', '03', '20260326_L1.nc',
    )
    assert os.path.exists(output_25)
    assert os.path.exists(output_26)
    _assert_netcdf_equal(
        output_25,
        os.path.join(TESTING_OUTPUT, 'IA', '2026', '03', '20260325_L1.nc'),
    )
    _assert_netcdf_equal(
        output_26,
        os.path.join(TESTING_OUTPUT, 'IA', '2026', '03', '20260326_L1.nc'),
    )


def test_process_raw_files_uses_previous_file_when_present(
        dirs,
        ceilometer, monkeypatch,
):
    _copy_input_fixtures(dirs['input_dir'])

    def _fake_glob(file_date, prefix):
        if file_date == date(2026, 3, 26):
            return [
                os.path.join(dirs['input_dir'], 'live_20260325_235741.nc'),
                os.path.join(dirs['input_dir'], 'live_20260326_001241.nc'),
                os.path.join(dirs['input_dir'], 'live_20260326_002741.nc'),
            ]
        return []

    monkeypatch.setattr(ceilometer, 'glob_day_raw_data', _fake_glob)

    ret = ceilometer.process_raw_files(
        start_date=date(2026, 3, 26),
        end_date=date(2026, 3, 26),
    )

    assert ret == 0
    output_file = os.path.join(
        dirs['output_dir'], 'IA', '2026', '03', '20260326_L1.nc',
    )
    assert os.path.exists(output_file)
    _assert_netcdf_equal(
        output_file,
        os.path.join(TESTING_OUTPUT, 'IA', '2026', '03', '20260326_L1.nc'),
    )


def test_process_raw_files_no_next_day_files(dirs, ceilometer, monkeypatch):
    _copy_input_fixtures(dirs['input_dir'])

    def _fake_glob(file_date, prefix):
        if file_date == date(2026, 3, 25):
            return [os.path.join(dirs['input_dir'], 'live_20260325_235741.nc')]
        return []

    monkeypatch.setattr(ceilometer, 'glob_day_raw_data', _fake_glob)

    ret = ceilometer.process_raw_files(
        start_date=date(2026, 3, 25),
        end_date=date(2026, 3, 25),
    )

    assert ret == 0
    assert os.path.exists(
        os.path.join(dirs['output_dir'], 'IA', '2026', '03', '20260325_L1.nc'),
    )


def test_process_raw_files_jobs_gt_1(dirs, ceilometer):
    _copy_input_fixtures(dirs['input_dir'])

    ret = ceilometer.process_raw_files(
        start_date='2026-03-25',
        end_date=date(2026, 3, 26),
        jobs=2,
    )

    assert ret == 0
    assert os.path.exists(
        os.path.join(dirs['output_dir'], 'IA', '2026', '03', '20260325_L1.nc'),
    )
    assert os.path.exists(
        os.path.join(dirs['output_dir'], 'IA', '2026', '03', '20260326_L1.nc'),
    )


# ---------------------------------------------------------------------------
# stratfinder_in_docker
# ---------------------------------------------------------------------------

def test_stratfinder_in_docker_rejects_relative_directory_mount():
    with pytest.raises(ValueError, match='directory_mount must be an absolute path'):
        Ceilometer.stratfinder_in_docker(
            today_file='today.nc',
            output_file='out.nc',
            beta_file='beta.nc',
            config_file='config.json',
            directory_mount='relative/path',
        )


def test_stratfinder_in_docker_builds_command_with_all_optional_args(
        tmp_path,
        monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    result = mock.MagicMock()
    result.returncode = 0

    with mock.patch(
        'ceilometer_toolbox.device.subprocess.run',
        return_value=result,
    ) as run:
        ret = Ceilometer.stratfinder_in_docker(
            today_file='today.nc',
            output_file='out.nc',
            beta_file='beta.nc',
            config_file='config.json',
            yesterday_file='yesterday.nc',
            overlap_file='overlap.nc',
            directory_mount=str(tmp_path),
            container_image='myimage:v1',
        )

    assert ret == 0
    cmd = run.call_args[0][0]
    assert '/data/yesterday.nc' in cmd
    assert '/data/overlap.nc' in cmd
    assert 'myimage:v1' in cmd


def test_stratfinder_in_docker_builds_command_without_optional_args(
        tmp_path,
        monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    result = mock.MagicMock()
    result.returncode = 42

    with mock.patch(
        'ceilometer_toolbox.device.subprocess.run',
        return_value=result,
    ) as run:
        ret = Ceilometer.stratfinder_in_docker(
            today_file='today.nc',
            output_file='out.nc',
            beta_file='beta.nc',
            config_file='config.json',
            directory_mount=str(tmp_path),
        )

    assert ret == 42
    cmd = run.call_args[0][0]
    assert repr('') in cmd


def test_stratfinder_in_docker_uses_cwd_when_no_directory_mount(monkeypatch):
    monkeypatch.chdir('/')
    result = mock.MagicMock()
    result.returncode = 0

    with mock.patch(
        'ceilometer_toolbox.device.subprocess.run',
        return_value=result,
    ) as run:
        Ceilometer.stratfinder_in_docker(
            today_file='today.nc',
            output_file='out.nc',
            beta_file='beta.nc',
            config_file='config.json',
        )

    cmd = run.call_args[0][0]
    assert '/:/data' in cmd


def test_stratfinder_in_docker_paths_are_above_directory_mount(tmp_path, monkeypatch):
    wd = tmp_path / 'data' / 'subdir'
    wd.mkdir(parents=True)
    monkeypatch.chdir(wd)

    with pytest.raises(
        ValueError,
        match='Input, output and config files must be located within the directory_mount',  # noqa: E501
    ):
        Ceilometer.stratfinder_in_docker(
            today_file='../today.nc',
            output_file='../out.nc',
            beta_file='../beta.nc',
            config_file='../config.json',
            # mount is subdir itself, so ../today.nc goes above it
            directory_mount=str(wd),
        )


def test_stratfinder_in_docker_paths_are_absolute_but_ok(tmp_path, monkeypatch):
    wd = tmp_path / 'data' / 'subdir'
    wd.mkdir(parents=True)
    monkeypatch.chdir(wd)

    with mock.patch(
        'ceilometer_toolbox.device.subprocess.run',
    ) as run:
        Ceilometer.stratfinder_in_docker(
            today_file=os.path.join(str(wd), 'today.nc'),
            output_file=os.path.join(str(wd), 'out.nc'),
            beta_file=os.path.join(str(wd), 'beta.nc'),
            config_file=os.path.join(str(wd), 'config.json'),
            # we are in data but configs are below but abspaths
            directory_mount=str(wd.parent),
        )

    cmd = run.call_args[0][0]
    assert '/data/subdir/config.json' in cmd


def test_stratfinder_in_docker_paths_are_absolute_but_parent_of_mount(
        tmp_path,
        monkeypatch,
):
    wd = tmp_path / 'data' / 'subdir'
    wd.mkdir(parents=True)
    monkeypatch.chdir(wd)

    with pytest.raises(
        ValueError,
        match='Input, output and config files must be located within the directory_mount',  # noqa: E501
    ):
        Ceilometer.stratfinder_in_docker(
            today_file=os.path.join(str(wd.parent), 'today.nc'),
            output_file=os.path.join(str(wd.parent), 'out.nc'),
            beta_file=os.path.join(str(wd.parent), 'beta.nc'),
            config_file=os.path.join(str(wd.parent), 'config.json'),
            # absolute paths in parent of mount → outside the mounted volume
            directory_mount=str(wd),
        )

# ---------------------------------------------------------------------------
# stratfinder_local
# ---------------------------------------------------------------------------


def test_stratfinder_local_builds_command_with_all_optional_args():
    result = mock.MagicMock()
    result.returncode = 0

    with mock.patch(
        'ceilometer_toolbox.device.subprocess.run',
        return_value=result,
    ) as run:
        ret = Ceilometer.stratfinder_local(
            executable_path='/usr/bin/stratfinder',
            today_file='today.nc',
            output_file='out.nc',
            beta_file='beta.nc',
            config_file='config.json',
            yesterday_file='yesterday.nc',
            overlap_file='overlap.nc',
        )

    assert ret == 0
    cmd = run.call_args[0][0]
    assert cmd[0] == '/usr/bin/stratfinder'
    assert 'yesterday.nc' in cmd
    assert 'overlap.nc' in cmd


def test_stratfinder_local_builds_command_without_optional_args():
    result = mock.MagicMock()
    result.returncode = 42

    with mock.patch(
        'ceilometer_toolbox.device.subprocess.run',
        return_value=result,
    ) as run:
        ret = Ceilometer.stratfinder_local(
            executable_path='/usr/bin/stratfinder',
            today_file='today.nc',
            output_file='out.nc',
            beta_file='beta.nc',
            config_file='config.json',
        )

    assert ret == 42
    cmd = run.call_args[0][0]
    assert repr('') in cmd


# ---------------------------------------------------------------------------
# process_l1_files
# ---------------------------------------------------------------------------

def test_process_l1_files_requires_config_file(dirs, archive):
    cel = Ceilometer(
        device_id='IA', input_dir=dirs['input_dir'], archive=archive,
    )
    with pytest.raises(ValueError, match='config_file must be provided'):
        cel.process_l1_files(start_date=date(2026, 3, 25))


def test_process_l1_files_skips_missing_today_file(dirs, ceilometer):
    with freeze_time('2026-03-25 12:00:00'):
        ret = ceilometer.process_l1_files(
            start_date=date(2026, 3, 25),
            config_file='config.json',
        )

    assert ret == 0


def test_process_l1_files_uses_str_start_and_end_dates(dirs, ceilometer):
    ret = ceilometer.process_l1_files(
        start_date='2026-03-25',
        end_date='2026-03-24',
        config_file='config.json',
    )
    assert ret == 0


def test_process_l1_files_uses_directory_mount_default(dirs, archive, monkeypatch):
    monkeypatch.chdir('/')
    cel = Ceilometer(
        device_id='IA',
        input_dir=dirs['input_dir'],
        archive=archive,
        stratfinder_config_file='config.json',
    )

    l1_path = archive.put_file('IA', 'L1', '2026-03-25')
    _touch(l1_path)

    mock_result = mock.MagicMock()
    mock_result.returncode = 0

    with mock.patch(
        'ceilometer_toolbox.device.subprocess.run',
        return_value=mock_result,
    ) as run:
        ret = cel.process_l1_files(
            start_date=date(2026, 3, 25),
            end_date=date(2026, 3, 25),
        )

    assert ret == 0
    cmd = run.call_args[0][0]
    assert '/:/data' in cmd


def test_process_l1_files_local_requires_executable_path(dirs, archive):
    cel = Ceilometer(
        device_id='IA',
        input_dir=dirs['input_dir'],
        archive=archive,
        stratfinder_config_file='config.json',
    )

    l1_path = archive.put_file('IA', 'L1', '2026-03-25')
    _touch(l1_path)

    with pytest.raises(ValueError, match='executable_path must be provided'):
        cel.process_l1_files(
            start_date=date(2026, 3, 25),
            end_date=date(2026, 3, 25),
            in_docker=False,
            directory_mount='/',
        )


def test_process_l1_files_local_runs_stratfinder_local(dirs, archive):
    cel = Ceilometer(
        device_id='IA',
        input_dir=dirs['input_dir'],
        archive=archive,
        stratfinder_config_file='config.json',
    )

    l1_path = archive.put_file('IA', 'L1', '2026-03-25')
    _touch(l1_path)

    mock_result = mock.MagicMock()
    mock_result.returncode = 0

    with mock.patch(
        'ceilometer_toolbox.device.subprocess.run',
        return_value=mock_result,
    ) as run:
        ret = cel.process_l1_files(
            start_date=date(2026, 3, 25),
            end_date=date(2026, 3, 25),
            in_docker=False,
            executable_path='/usr/bin/stratfinder',
            directory_mount='/',
        )

    assert ret == 0
    cmd = run.call_args[0][0]
    assert cmd[0] == '/usr/bin/stratfinder'


def test_process_l1_files_raises_on_stratfinder_failure(dirs, archive):
    cel = Ceilometer(
        device_id='IA',
        input_dir=dirs['input_dir'],
        archive=archive,
        stratfinder_config_file='config.json',
    )

    l1_path = archive.put_file('IA', 'L1', '2026-03-25')
    _touch(l1_path)

    mock_result = mock.MagicMock()
    mock_result.returncode = 1

    with mock.patch(
        'ceilometer_toolbox.device.subprocess.run',
        return_value=mock_result,
    ):
        with pytest.raises(RuntimeError, match='Stratfinder failed'):
            cel.process_l1_files(
                start_date=date(2026, 3, 25),
                end_date=date(2026, 3, 25),
                directory_mount='/',
            )


# ---------------------------------------------------------------------------
# process_stratfinder_qc
# ---------------------------------------------------------------------------

def test_process_stratfinder_qc_requires_all_config_files(dirs, archive):
    cel = Ceilometer(
        device_id='IA', input_dir=dirs['input_dir'], archive=archive,
    )
    with pytest.raises(ValueError, match='config_file, value_config_file'):
        cel.process_stratfinder_qc(start_date=date(2026, 3, 25))


def test_process_stratfinder_qc_requires_all_configs_partially_set(dirs, archive):
    cel = Ceilometer(
        device_id='IA',
        input_dir=dirs['input_dir'],
        archive=archive,
        stratfinder_config_file='config.json',
    )
    with pytest.raises(ValueError, match='config_file, value_config_file'):
        cel.process_stratfinder_qc(start_date=date(2026, 3, 25))


def test_process_stratfinder_qc_skips_when_files_missing(dirs, archive):
    cel = Ceilometer(
        device_id='IA',
        input_dir=dirs['input_dir'],
        archive=archive,
        stratfinder_config_file='config.json',
        stratfinder_qc_value_config_file='values.toml',
        stratfinder_qc_metadata_file='meta.toml',
    )

    with freeze_time('2026-03-25 12:00:00'):
        ret = cel.process_stratfinder_qc(start_date=date(2026, 3, 25))

    assert ret == 0


def test_process_stratfinder_qc_uses_str_start_and_end_dates(dirs, archive):
    cel = Ceilometer(
        device_id='IA',
        input_dir=dirs['input_dir'],
        archive=archive,
        stratfinder_config_file='config.json',
        stratfinder_qc_value_config_file='values.toml',
        stratfinder_qc_metadata_file='meta.toml',
    )

    ret = cel.process_stratfinder_qc(
        start_date='2026-03-25',
        end_date='2026-03-24',
    )

    assert ret == 0


def test_process_stratfinder_qc_runs_and_breaks_on_failure(dirs, archive):
    cel = Ceilometer(
        device_id='IA',
        input_dir=dirs['input_dir'],
        archive=archive,
        stratfinder_config_file='config.json',
        stratfinder_qc_value_config_file='values.toml',
        stratfinder_qc_metadata_file='meta.toml',
    )

    yesterday = archive.put_file('IA', 'L2A_stratfinder', '2026-03-24')
    today = archive.put_file('IA', 'L2A_stratfinder', '2026-03-25')
    _touch(yesterday)
    _touch(today)

    with mock.patch('ceilometer_toolbox.device.qc_daily_final', return_value=1):
        ret = cel.process_stratfinder_qc(
            start_date=date(2026, 3, 25),
            end_date=date(2026, 3, 26),
        )

    assert ret == 1


def test_process_stratfinder_qc_with_explicit_config_files(dirs, archive):
    # Exercises the False branches of the three `if not config_file:` guards
    cel = Ceilometer(
        device_id='IA', input_dir=dirs['input_dir'], archive=archive,
    )

    yesterday = archive.put_file('IA', 'L2A_stratfinder', '2026-03-24')
    today = archive.put_file('IA', 'L2A_stratfinder', '2026-03-25')
    _touch(yesterday)
    _touch(today)

    with mock.patch('ceilometer_toolbox.device.qc_daily_final', return_value=0):
        ret = cel.process_stratfinder_qc(
            start_date=date(2026, 3, 25),
            end_date=date(2026, 3, 25),
            config_file='config.json',
            value_config_file='values.toml',
            stratfinder_metadata_file='meta.toml',
        )

    assert ret == 0


def test_process_stratfinder_qc_succeeds(dirs, archive):
    cel = Ceilometer(
        device_id='IA',
        input_dir=dirs['input_dir'],
        archive=archive,
        stratfinder_config_file='config.json',
        stratfinder_qc_value_config_file='values.toml',
        stratfinder_qc_metadata_file='meta.toml',
    )

    yesterday = archive.put_file('IA', 'L2A_stratfinder', '2026-03-24')
    today = archive.put_file('IA', 'L2A_stratfinder', '2026-03-25')
    _touch(yesterday)
    _touch(today)

    with mock.patch('ceilometer_toolbox.device.qc_daily_final', return_value=0):
        ret = cel.process_stratfinder_qc(
            start_date=date(2026, 3, 25),
            end_date=date(2026, 3, 25),
        )

    assert ret == 0


# ---------------------------------------------------------------------------
# Multi-device / non-'IA' station tests
# ---------------------------------------------------------------------------

def test_process_raw_files_non_default_device_id(dirs, archive):
    # Verify output is stored under the correct device directory
    _copy_input_fixtures(dirs['input_dir'])
    cel = Ceilometer(
        device_id='IB',
        input_dir=dirs['input_dir'],
        archive=archive,
        raw2l1_config_file=RAW2L1_CONF,
    )

    ret = cel.process_raw_files(
        start_date=date(2026, 3, 25),
        end_date=date(2026, 3, 25),
    )

    assert ret == 0
    ib_output = os.path.join(
        dirs['output_dir'], 'IB', '2026', '03', '20260325_L1.nc',
    )
    ia_output = os.path.join(
        dirs['output_dir'], 'IA', '2026', '03', '20260325_L1.nc',
    )
    assert os.path.exists(ib_output)
    assert not os.path.exists(ia_output)


def test_process_l1_files_yesterday_sourced_from_own_device(dirs, archive):
    # Regression test: yesterday's L1 must be looked up under self.device_id,
    # not a hardcoded 'IA'. Only IB files exist; if the lookup used 'IA' the
    # yesterday_file would be None and cmd[-2] would be repr('').
    cel = Ceilometer(
        device_id='IB',
        input_dir=dirs['input_dir'],
        archive=archive,
        stratfinder_config_file='config.json',
    )

    today_path = archive.put_file('IB', 'L1', '2026-03-25')
    yesterday_path = archive.put_file('IB', 'L1', '2026-03-24')
    _touch(today_path)
    _touch(yesterday_path)

    mock_result = mock.MagicMock()
    mock_result.returncode = 0

    with mock.patch(
        'ceilometer_toolbox.device.subprocess.run',
        return_value=mock_result,
    ) as run:
        ret = cel.process_l1_files(
            start_date=date(2026, 3, 25),
            end_date=date(2026, 3, 25),
            directory_mount='/',
        )

    assert ret == 0
    cmd = run.call_args[0][0]
    # cmd[-2] is the yesterday_file argument; repr('') means it was not found
    assert cmd[-2] != repr(''), 'yesterday_file was not found — device_id bug regressed'


def test_process_stratfinder_qc_non_default_device_id(dirs, archive):
    cel = Ceilometer(
        device_id='IB',
        input_dir=dirs['input_dir'],
        archive=archive,
        stratfinder_config_file='config.json',
        stratfinder_qc_value_config_file='values.toml',
        stratfinder_qc_metadata_file='meta.toml',
    )

    yesterday = archive.put_file('IB', 'L2A_stratfinder', '2026-03-24')
    today = archive.put_file('IB', 'L2A_stratfinder', '2026-03-25')
    _touch(yesterday)
    _touch(today)

    with mock.patch('ceilometer_toolbox.device.qc_daily_final', return_value=0):
        ret = cel.process_stratfinder_qc(
            start_date=date(2026, 3, 25),
            end_date=date(2026, 3, 25),
        )

    assert ret == 0
    assert os.path.exists(
        os.path.join(
            dirs['output_dir'], 'IB', '2026', '03', '20260325_L2B_stratfinder.nc',
        ),
    )


def test_two_devices_in_same_archive_are_independent(dirs, archive):
    # Both devices live in the same archive; operations on one must not affect
    # the other.
    cel_ia = Ceilometer(
        device_id='IA',
        input_dir=dirs['input_dir'],
        archive=archive,
        raw2l1_config_file=RAW2L1_CONF,
    )
    cel_ib = Ceilometer(
        device_id='IB',
        input_dir=dirs['input_dir'],
        archive=archive,
        raw2l1_config_file=RAW2L1_CONF,
    )

    _copy_input_fixtures(dirs['input_dir'])

    cel_ia.process_raw_files(
        start_date=date(
            2026, 3, 25,
        ), end_date=date(2026, 3, 25),
    )
    cel_ib.process_raw_files(
        start_date=date(
            2026, 3, 25,
        ), end_date=date(2026, 3, 25),
    )

    ia_files = archive.get_files(device_id='IA', file_type='L1')
    ib_files = archive.get_files(device_id='IB', file_type='L1')
    assert len(ia_files) == 1
    assert len(ib_files) == 1
    assert 'IA' in ia_files[0]
    assert 'IB' in ib_files[0]
    # Deleting IA's file leaves IB's intact
    archive.delete_file('IA', 'L1', '2026-03-25')
    assert archive.get_file_or_none('IA', 'L1', '2026-03-25') is None
    assert archive.get_file_or_none('IB', 'L1', '2026-03-25') is not None
