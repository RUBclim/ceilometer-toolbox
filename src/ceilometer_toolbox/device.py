import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from datetime import date
from datetime import datetime
from datetime import timedelta
from functools import lru_cache
from glob import glob
from multiprocessing import Pool
from typing import Any
from typing import Literal

import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ceilometer_toolbox.data import atomic_write_path
from ceilometer_toolbox.data import CeilometerArchive
from ceilometer_toolbox.utils import add_solar_times
from ceilometer_toolbox.utils import LDR_CMAP
from ceilometer_toolbox.utils import resample_dataset
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qc_sf_python.qc_daily_final import qc_daily_final
from raw2l1.raw2l1 import raw2l1


@lru_cache(maxsize=None)
def _file_date_pattern(prefix: str) -> re.Pattern[str]:
    """Compile (and cache) the regex matching ``{prefix}YYYYMMDD_HHMMSS`` for the
    given prefix. Cached so we don't recompile on every filename parse."""
    return re.compile(rf"{re.escape(prefix)}(\d{{8}}_\d{{6}})")


class Ceilometer:
    """Class for processing ceilometer data and making plots."""

    def __init__(
            self,
            device_id: str,
            input_dir: str,
            archive: CeilometerArchive,
            raw2l1_config_file: str | None = None,
            stratfinder_config_file: str | None = None,
            stratfinder_qc_value_config_file: str | None = None,
            stratfinder_qc_metadata_file: str | None = None,
            calibration_profile_beta: xr.DataArray | None = None,
            calibration_profile_x_pol: xr.DataArray | None = None,
            calibration_profile_p_pol: xr.DataArray | None = None,
    ) -> None:
        """
        :param device_id: The ID of the ceilometer device to process. This should
            match the device_id used in the CeilometerArchive for storing the files.
        :param input_dir: The directory where the raw ceilometer files are stored. This
            should be a directory that contains the raw files with the naming
            convention live_YYYYMMDD_HHMMSS.nc, where YYYYMMDD is the date of the file
            and HHMMSS is the time of the file. The files should be organized in a way
            that allows globbing for a specific date
        :param archive: The CeilometerArchive instance to use for reading and writing
            files. This should be initialized with the same device_id as the one
            provided to this class.
        :param raw2l1_config_file: The path to the raw2l1 configuration file (.conf).
        :param stratfinder_config_file: The path to the stratfinder configuration
            file (.json).
        :param stratfinder_qc_value_config_file: The path to the stratfinder QC value
            config file (.toml).
        :param stratfinder_qc_metadata_file: The path to the stratfinder QC
            metadata (.toml)
        :param calibration_profile_beta: Median dark measurement profile
            subtracted from ``beta`` during L1 processing. When ``None`` no
            correction is applied. Typically obtained from
            :meth:`derive_median_dark_measurement_profile`.
        :param calibration_profile_x_pol: Same as ``calibration_profile_beta``
            but for the cross-polarization component (subtracted from
            ``rcs_2``). Only meaningful for polarization-capable devices.
        :param calibration_profile_p_pol: Same as ``calibration_profile_beta``
            but for the parallel-polarization component (subtracted from
            ``rcs_1``). Only meaningful for polarization-capable devices.
        """
        self.archive = archive
        self.input_dir = input_dir
        self.device_id = device_id
        self.raw2l1_config_file = raw2l1_config_file
        self.stratfinder_config_file = stratfinder_config_file
        self.stratfinder_qc_value_config_file = stratfinder_qc_value_config_file
        self.stratfinder_qc_metadata_file = stratfinder_qc_metadata_file
        self.calibration_profile_beta = calibration_profile_beta
        self.calibration_profile_x_pol = calibration_profile_x_pol
        self.calibration_profile_p_pol = calibration_profile_p_pol

    def glob_day_raw_data(self, file_date: date, prefix: str) -> list[str]:
        """Glob the raw ceilometer files for a given date and prefix.

        :param file_date: The date of the files to glob.
        :param prefix: The prefix of the files to glob. This is usually ``live_`` for
            raw files, but may be different if the naming convention is different.
            The glob pattern is ``{prefix}{file_date:%Y%m%d}_*.nc``.
        :return: list of matching file paths (unsorted)
        """
        return glob(
            os.path.join(
                self.input_dir,
                f"{prefix}{file_date:%Y%m%d}_*.nc",
            ),
        )

    def _parse_file_date_from_name(self, file_name: str, prefix: str) -> datetime:
        """Parse the date and time from a raw file name.

        :param file_name: The name of the file to parse. This should be the full path
            to the file, but only the file name will be used for parsing.
        :param prefix: The prefix of the files. This is usually ``live_`` for raw
            files, but may be different if the naming convention is different.
        :return: The parsed date and time as a datetime object.
        :raises ValueError: if the file name does not match the expected
            ``{prefix}YYYYMMDD_HHMMSS`` pattern.
        """
        match = _file_date_pattern(prefix).search(file_name)
        if match is None:
            raise ValueError(
                f'Could not parse date from file name: {file_name!r} '
                f'with prefix {prefix!r}',
            )
        return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')

    def get_raw_files(self, start: datetime, end: datetime, prefix: str) -> list[str]:
        # let's first glob to the closest common pattern
        common_part = os.path.commonprefix([
            f'{prefix}{start:%Y%m%d_%H%M%S}',
            f'{prefix}{end:%Y%m%d_%H%M%S}',
        ])
        globbed_files = sorted(glob(os.path.join(self.input_dir, f"{common_part}*.nc")))
        final_files = []
        # now select the files precisely based on the name
        for f in globbed_files:
            f_date = self._parse_file_date_from_name(f, prefix=prefix)
            # we need one more file after the last one since the last part will
            # be written after the cut off date
            if start <= f_date <= end:
                final_files.append(f)

        if not final_files:
            raise ValueError(
                f'No files found between {start} and {end} with prefix {prefix!r}',
            )
        # get the index of the last file included
        idx_last_file = globbed_files.index(final_files[-1]) + 1
        if idx_last_file < len(globbed_files):
            final_files.append(globbed_files[idx_last_file])
        return final_files

    def to_l1(
            self,
            file_date: date,
            input_files: str | list[str],
            output_file: str,
            config_file: str | None = None,
            ancillary_files: str | list[str] = [],
            min_file_size: int = 0,
            check_timeliness: bool = False,
            filter_max_age: int = 2,
            filter_day: bool = False,
            log_file: str | None = None,
            log_level: str = 'info',
            verbose: str = 'info',
            calibration_profile_beta: xr.DataArray | None = None,
            calibration_profile_x_pol: xr.DataArray | None = None,
            calibration_profile_p_pol: xr.DataArray | None = None,
    ) -> int:
        """Convert raw ceilometer files to level 1 using the raw2l1 tool.

        :param file_date: The date of the files to process
        :param config_file: The path to the raw2l1 configuration file
        :param input_files: The raw files to process, can be a single file or a list of
            files
        :param output_file: The path to the output file
        :param ancillary_files: The ancillary files to use, can be a single file or a
            list of files
        :param min_file_size: The minimum size of input file in bytes. Files with a
            smaller size will be rejected.
        :param check_timeliness: Check if the data read are not to old or in the
            future. By default it checks thats data have a maximum age of 2 hours.
            This value can be changed with option ``file_max_age``.
        :param filter_max_age: Allow to define the maximum age of data in a file in
            hours
        :param filter_day: Only keep data of date provided as arguments
        :param log_file: File where logs will be saved
        :param log_level: Level of logs store in the log file. Choices are debug, info,
            warning, error, critical
        :param verbose: Level of verbose in the terminal. Same choices as log_level
        :param calibration_profile_beta: Override for
            ``self.calibration_profile_beta``. When the resolved profile is not
            ``None`` it is subtracted from ``beta`` after raw2l1 has run.
        :param calibration_profile_x_pol: Override for
            ``self.calibration_profile_x_pol``. Subtracted from ``rcs_2`` when
            present.
        :param calibration_profile_p_pol: Override for
            ``self.calibration_profile_p_pol``. Subtracted from ``rcs_1`` when
            present.

        :return: The return code of the raw2l1 tool, 0 if successful, non-zero otherwise
        """
        if calibration_profile_beta is None:
            calibration_profile_beta = self.calibration_profile_beta
        if calibration_profile_x_pol is None:
            calibration_profile_x_pol = self.calibration_profile_x_pol
        if calibration_profile_p_pol is None:
            calibration_profile_p_pol = self.calibration_profile_p_pol
        if not config_file:
            config_file = self.raw2l1_config_file

        if config_file is None:
            raise ValueError(
                'config_file must be provided either in the method call or in the '
                'class initialization',
            )

        # build the correct command line arguments for raw2l1
        input_files = (
            [input_files]
            if isinstance(input_files, str)
            else input_files
        )
        ancillary_files = (
            [ancillary_files]
            if isinstance(ancillary_files, str)
            else ancillary_files
        )
        # add prefix argument and flatten
        ancillary_files = [
            item for anc in ancillary_files for item in ['--ancillary', anc]
        ]
        if log_file is None:
            log_file = os.path.join(
                tempfile.gettempdir(),
                f"raw2l1_{file_date:%Y%m%d}.log",
            )

        with atomic_write_path(final_path=output_file, override=True) as tmp_file:
            cmd = [
                file_date.strftime('%Y%m%d'),
                config_file,
                *input_files,
                tmp_file,
                *ancillary_files,
                '-file_min_size',
                str(min_file_size),
                '--check_timeliness' if check_timeliness else '',
                '-file_max_age',
                str(filter_max_age),
                '--filter-day' if filter_day else '',
                '-log',
                log_file,
                '-log_level',
                log_level.lower(),
                '-v',
                verbose.lower(),
            ]
            # now clean up the unset optional arguments
            cmd = [arg for arg in cmd if arg != '']
            ret = raw2l1(cmd)
            if ret != 0:
                raise RuntimeError(
                    f"raw2l1 failed with return code {ret}, "
                    f"see log file {log_file} for details",
                )
            any_calibration = any(
                p is not None
                for p in (
                    calibration_profile_beta,
                    calibration_profile_x_pol,
                    calibration_profile_p_pol,
                )
            )
            if any_calibration:
                # TODO: this creates some IO overhead
                with xr.open_dataset(tmp_file) as ds:
                    ds = ds.load()

                original_encoding = {
                    name: var.encoding.copy()
                    for name, var in ds.variables.items()
                }
                packable_keys = (
                    'zlib', 'complevel', 'shuffle', 'chunksizes', 'dtype', '_FillValue',
                    'missing_value',
                )

                # Mapping from raw-file variable name (which is what the
                # calibration profile was derived against) to its corresponding
                # L1 variable name written by raw2l1.
                calibrations = (
                    ('beta', calibration_profile_beta),
                    ('rcs_2', calibration_profile_x_pol),
                    ('rcs_1', calibration_profile_p_pol),
                )
                rcs_calibrated = set()
                for l1_var, profile in calibrations:
                    if profile is None or l1_var not in ds:
                        continue

                    # NaN / inf in the profile mean "no information for this
                    # gate" - treat as a zero correction.
                    # the eprofile.ini has some values marked as $double$ (64 bit),
                    # however, the CL61 only reports in 32bit, hence 32bit values
                    # are saved in 64 bit variables. In the original file this is not
                    # a problem, but when we subtract the profile from it, the result
                    # seems to make use of the higher 64 bit precision and doubles the
                    # size of the column
                    correction = profile.where(np.isfinite(profile), 0).astype('f4')
                    # this makes it preserve the attributes, keep_attrs=True does not
                    # work here.
                    ds.update(
                        other={l1_var: (ds[l1_var] - correction.values).astype('f4')},
                    )
                    rcs_calibrated.add(l1_var)

                if rcs_calibrated & {'rcs_1', 'rcs_2'} and {'rcs_1', 'rcs_2'} <= set(ds):  # noqa: E501
                    ds['linear_depol_ratio'].values = ds['rcs_2'].values / ds['rcs_1'].values  # noqa: E501
                    ds['rcs_0'].values = ds['rcs_1'].values + ds['rcs_2'].values
                # write the calibrated dataset back to the original output file
                # we need to use the same settings as in raw2l1 to match it closely
                for v in ('beta', 'rcs_1', 'rcs_2', 'ldr', 'rcs_0'):
                    if v in ds and v in original_encoding:
                        v_enc = {
                            k: original_encoding[v][k]
                            for k in packable_keys if k in original_encoding[v]
                        }
                        ds[v].encoding = v_enc

                ds.to_netcdf(tmp_file, mode='w')
                # now let the atomic write happen
            return ret

    def process_raw_files(
            self,
            start_date: date | str | None = None,
            end_date: date | str | None = None,
            prefix: str = 'live_',
            jobs: int = 1,
            config_file: str | None = None,
            calibration_profile_beta: xr.DataArray | None = None,
            calibration_profile_x_pol: xr.DataArray | None = None,
            calibration_profile_p_pol: xr.DataArray | None = None,
    ) -> int:
        """Process raw ceilometer files since a given date and convert them to level 1.

        :param start_date: The date to start processing from. This can be a date object
            or a string in the format YYYY-MM-DD. If None, processing will start from
            the most recently processed L1 date already in the archive (defaults to
            1970-01-01 if no L1 files exist yet).
        :param end_date: The date to stop processing at. This can be a date object or a
            string in the format YYYY-MM-DD. If None, processing will continue until the
            current date.
        :param prefix: The prefix of the raw files to process. This is usually ``live_``
        :param jobs: The number of parallel processes to use for processing the files.
        :param config_file: Option to override the raw2l1 configuration file provided
            in the class initialization.
        :param calibration_profile_beta: Override for
            ``self.calibration_profile_beta`` for this call. See :meth:`to_l1`.
        :param calibration_profile_x_pol: Override for
            ``self.calibration_profile_x_pol`` for this call.
        :param calibration_profile_p_pol: Override for
            ``self.calibration_profile_p_pol`` for this call.
        """
        if calibration_profile_beta is None:
            calibration_profile_beta = self.calibration_profile_beta
        if calibration_profile_x_pol is None:
            calibration_profile_x_pol = self.calibration_profile_x_pol
        if calibration_profile_p_pol is None:
            calibration_profile_p_pol = self.calibration_profile_p_pol
        if not config_file:
            config_file = self.raw2l1_config_file

        if config_file is None:
            raise ValueError(
                'config_file must be provided either in the method call or in the '
                'class initialization',
            )

        if start_date is None:
            start_date = self.archive.latest_date(
                device_id=self.device_id,
                file_type='L1',
            ) or date(1970, 1, 1)

        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)

        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        if end_date is not None and start_date > end_date:
            raise ValueError('start_date cannot be after end_date')

        end_date = end_date if end_date is not None else date.today()

        ret = 0
        tasks = []
        while start_date <= end_date:
            # compile the file pattern for the current date
            # We require at least one file from the current day. The first file of
            # the next day is optional but helps closing out the current day.
            current_day = self.glob_day_raw_data(start_date, prefix=prefix)
            next_day = self.glob_day_raw_data(
                start_date + timedelta(days=1),
                prefix=prefix,
            )
            if not current_day:
                print(f"No files found for current day {start_date}, skipping")
                start_date += timedelta(days=1)
                continue

            files = current_day + next_day

            # now find the files the we actually have to pass to the tool
            files = sorted(files)
            first_file = files[0]
            last_file = files[-1]
            # now find the index of the first file of the current day
            for idx, file in enumerate(files):  # pragma: no branch
                if f"{prefix}{start_date:%Y%m%d}" in file:
                    if idx == 0:
                        # the first file is already from the current day, so we can
                        # start from there
                        first_file = file
                    else:
                        # we have a file from the previous day, so we need to start
                        # from there
                        first_file = files[idx - 1]
                    break

            # now find the index of the last file of the current day
            for idx, file in enumerate(files[idx:]):
                if f"{prefix}{start_date:%Y%m%d}" not in file:
                    last_file = file
                    break
            else:
                # we didn't find a file from the next day, so we can end with the
                # last file we have
                last_file = files[-1]

            files_to_process = files[
                files.index(
                    first_file,
                ): files.index(last_file) + 1
            ]
            # process this in multiple processes
            kwargs = {
                'file_date': start_date,
                'input_files': files_to_process,
                'config_file': config_file,
                'output_file': self.archive.put_file(
                    device_id=self.device_id,
                    file_type='L1',
                    file_date=start_date,
                    override=True,
                ),
                'filter_day': True,
                'log_level': 'info',
                'calibration_profile_beta': calibration_profile_beta,
                'calibration_profile_x_pol': calibration_profile_x_pol,
                'calibration_profile_p_pol': calibration_profile_p_pol,
            }
            if jobs > 1:
                tasks.append(kwargs)

            else:
                ret |= self.to_l1(**kwargs)  # type: ignore[arg-type]

            start_date += timedelta(days=1)

        if jobs > 1 and tasks:
            with Pool(processes=jobs) as pool:
                task_args: list[tuple[Any, ...]] = [
                    (
                        task['file_date'],
                        task['input_files'],
                        task['output_file'],
                        task['config_file'],
                        [],
                        0,
                        False,
                        2,
                        task['filter_day'],
                        None,
                        task['log_level'],
                        'info',
                        task['calibration_profile_beta'],
                        task['calibration_profile_x_pol'],
                        task['calibration_profile_p_pol'],
                    )
                    for task in tasks
                ]
                results = pool.starmap(self.to_l1, task_args)
                ret |= sum(results)

        return ret

    @staticmethod
    def stratfinder_in_docker(
            today_file: str,
            output_file: str,
            beta_file: str,
            config_file: str,
            yesterday_file: str | None = None,
            overlap_file: str | None = None,
            container_image: str = 'ghcr.io/rubclim/stratfinder:latest',
            directory_mount: str | None = None,
    ) -> int:
        """Run the stratfinder algorithm in a Docker container. This cannot be run in
            parallel since it depends on the output of the previous day.

        This is necessary because the stratfinder algorithm is implemented in
        Matlab and requires the Matlab Runtime to run.

        :param config_file: The path to the stratfinder configuration file (json)
        :param today_file: The path to the input file for the current day to
            process. This should be a L1 file output from the raw2l1 tool.
        :param output_file: Path to the output file for the stratfinder results.
        :param beta_file: The path to the output file for the beta results
            outputted by stratfinder.
        :param yesterday_file: The path to the input file for the previous day to
            process. This should be a L1 file output from the raw2l1 tool.
        :param overlap_file: The path to the input file for the overlap correction.
            This can be omitted if no overlap correction is desired.
        :param container_image: The name of the Docker image to use for running
            stratfinder. Please see: https://github.com/RUBclim/STRATfinder-docker
        :param directory_mount: The directory to mount in the Docker container.
            This should be an absolute path. If None, the current working directory
            will be used. The input and output files should be located in this
            directory or its subdirectories.
        """
        local_dir = directory_mount if directory_mount is not None else os.getcwd()

        if not os.path.isabs(local_dir):
            raise ValueError('directory_mount must be an absolute path')

        def _to_container_path(path: str) -> str:
            abs_path = os.path.abspath(path)
            rel = os.path.relpath(abs_path, local_dir)
            if rel.startswith('..'):
                raise ValueError(
                    f'Input, output and config files must be located within the '
                    f'directory_mount or its subdirectories. Moving above the mounted '
                    f'directory is not possible. If this is needed, change your '
                    f'directory_mount to a higher level directory that includes all '
                    f'needed files. Offending path: {path!r}, relative path: {rel!r}',
                )
            return os.path.join('/data', rel)

        today_file = _to_container_path(today_file)
        output_file = _to_container_path(output_file)
        beta_file = _to_container_path(beta_file)
        yesterday_file = _to_container_path(yesterday_file) if yesterday_file else None
        overlap_file = _to_container_path(overlap_file) if overlap_file else None
        dyn_config = _to_container_path(config_file)

        cmd = (
            'docker',
            'run',
            '-e',
            'AGREE_TO_MATLAB_RUNTIME_LICENSE=yes',
            # use the current user to get permissions right in the folder
            '-u',
            f"{os.getuid()}:{os.getgid()}",
            '--rm',
            '--workdir',
            '/data',
            '-v',
            f"{local_dir}:/data",
            container_image,
            dyn_config,
            overlap_file or repr(''),
            today_file,
            output_file,
            beta_file,
            yesterday_file or repr(''),
            repr(''),
        )
        result = subprocess.run(cmd, stdout=None, stderr=None)
        return result.returncode

    @staticmethod
    def stratfinder_local(
            executable_path: str,
            today_file: str,
            output_file: str,
            beta_file: str,
            config_file: str,
            yesterday_file: str | None = None,
            overlap_file: str | None = None,
    ) -> int:
        """Run the stratfinder algorithm locally. This cannot be run in parallel since
        it depends on the output of the previous day.

        :param executable_path: The path to the stratfinder executable. This should be
            the bash script that is provided along with the stratfinder Matlab
            distribution.
        :param config_file: The path to the stratfinder configuration file (json)
        :param today_file: The path to the input file for the current day to
            process. This should be a L1 file output from the raw2l1 tool.
        :param output_file: Path to the output file for the stratfinder results.
        :param beta_file: The path to the output file for the beta results
            outputted by stratfinder.
        :param yesterday_file: The path to the input file for the previous day to
            process. This should be a L1 file output from the raw2l1 tool.
        :param overlap_file: The path to the input file for the overlap correction.
            This can be omitted if no overlap correction is desired.
        """
        cmd = (
            executable_path,
            config_file,
            overlap_file or repr(''),
            today_file,
            output_file,
            beta_file,
            yesterday_file or repr(''),
            repr(''),
        )
        result = subprocess.run(cmd, stdout=None, stderr=None)
        return result.returncode

    def process_l1_files(
            self,
            start_date: date | str | None = None,
            end_date: date | str | None = None,
            config_file: str | None = None,
            directory_mount: str | None = None,
            in_docker: bool = True,
            executable_path: str | None = None,
            overlap_file: str | None = None,
    ) -> int:
        """Process the L1 files for the given date and all subsequent dates
        until end_date using the stratfinder algorithm.

        :param start_date: The date to start processing from.
        :param end_date: The date to stop processing at. If None, processing will
            continue until the current date.
        :param config_file: The path to the stratfinder configuration file (json).
        :param directory_mount: The directory to mount in the Docker container.
        :param in_docker: Whether to run stratfinder in a Docker container or use a
            local executable.
        :param executable_path: The path to the local stratfinder executable. This is
            only used if in_docker is False. This should be the bash script that is
            provided along with the stratfinder Matlab distribution.
        :param overlap_file: The path to the input file for the overlap correction.
            This can be omitted if no overlap correction is desired.
        """
        if start_date is None:
            start_date_beta = self.archive.latest_date(
                device_id=self.device_id,
                file_type='L2A_beta',
            ) or date(1970, 1, 1)
            start_date_strat = self.archive.latest_date(
                device_id=self.device_id,
                file_type='L2A_stratfinder',
            ) or date(1970, 1, 1)
            start_date = min(start_date_beta, start_date_strat)

        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)

        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        if not config_file:
            config_file = self.stratfinder_config_file

        if config_file is None:
            raise ValueError(
                'config_file must be provided either in the method call or in the '
                'class initialization',
            )

        if directory_mount is None:
            directory_mount = os.getcwd()

        end = end_date if end_date is not None else date.today()
        ret = 0
        while start_date <= end:
            today_file = self.archive.get_file_or_none(
                device_id=self.device_id,
                file_type='L1',
                file_date=start_date,
            )
            if today_file is None:
                print(f"File {today_file} does not exist, skipping")
                start_date += timedelta(days=1)
                continue

            yesterday = start_date - timedelta(days=1)
            yesterday_file = self.archive.get_file_or_none(
                device_id=self.device_id,
                file_type='L1',
                file_date=yesterday,
            )
            with (
                self.archive.atomic_put_file(
                    device_id=self.device_id,
                    file_type='L2A_stratfinder',
                    file_date=start_date,
                    override=True,
                ) as output_file,
                self.archive.atomic_put_file(
                    device_id=self.device_id,
                    file_type='L2A_beta',
                    file_date=start_date,
                    override=True,
                ) as beta_file,
            ):
                if in_docker:
                    ret = self.stratfinder_in_docker(
                        config_file=config_file,
                        today_file=today_file,
                        output_file=output_file,
                        beta_file=beta_file,
                        yesterday_file=yesterday_file,
                        overlap_file=overlap_file,
                        directory_mount=directory_mount,
                    )
                else:
                    if not executable_path:
                        raise ValueError(
                            'executable_path must be provided if in_docker is False',
                        )
                    ret = self.stratfinder_local(
                        executable_path=executable_path,
                        config_file=config_file,
                        today_file=today_file,
                        output_file=output_file,
                        beta_file=beta_file,
                        yesterday_file=yesterday_file,
                        overlap_file=overlap_file,
                    )
                if ret != 0:
                    print(f"Stratfinder failed for {start_date}, stopping")
                    raise RuntimeError(
                        f"Stratfinder failed for {start_date}. Exit code: {ret}",
                    )

                start_date += timedelta(days=1)

        return ret

    def process_stratfinder_qc(
            self,
            start_date: date | str | None = None,
            end_date: date | str | None = None,
            config_file: str | None = None,
            value_config_file: str | None = None,
            stratfinder_metadata_file: str | None = None,
    ) -> int:
        """Process the stratfinder output files for the given date and all subsequent
            dates until end_date using the stratfinder QC algorithm.

        This cannot be run in parallel since it depends on the output of the previous
        day.

        :param archive: The CeilometerArchive instance to use for reading and
            writing files.
        :param start_date: The date to start processing from.
        :param end_date: The date to stop processing at. If None, processing will
            continue until the current date.
        :param config_file: The path to the stratfinder QC config file (json).
        :param value_config_file: The path to the value config file (toml)
            for the stratfinder QC.
        :param stratfinder_metadata_file: The path to the stratfinder metadata
            file (toml) for the stratfinder QC.
        """
        if not config_file:
            config_file = self.stratfinder_config_file
        if not value_config_file:
            value_config_file = self.stratfinder_qc_value_config_file
        if not stratfinder_metadata_file:
            stratfinder_metadata_file = self.stratfinder_qc_metadata_file

        if any(
            [
                config_file is None,
                value_config_file is None,
                stratfinder_metadata_file is None,
            ],
        ):
            raise ValueError(
                'config_file, value_config_file and stratfinder_metadata_file must be '
                'provided either in the method call or in the class initialization',
            )

        if start_date is None:
            start_date = self.archive.latest_date(
                device_id=self.device_id,
                file_type='L2B_stratfinder',
            ) or date(1970, 1, 1)

        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)

        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        end = end_date if end_date is not None else date.today()
        ret = 0
        while start_date <= end:
            yesterday_file = self.archive.get_file_or_none(
                device_id=self.device_id,
                file_type='L2A_stratfinder',
                file_date=start_date - timedelta(days=1),
            )
            today_file = self.archive.get_file_or_none(
                device_id=self.device_id,
                file_type='L2A_stratfinder',
                file_date=start_date,
            )
            tomorrow_file = self.archive.get_file_or_none(
                device_id=self.device_id,
                file_type='L2A_stratfinder',
                file_date=start_date + timedelta(days=1),
            )
            if not today_file:
                print(f"The today file for {start_date} does not exist")
                start_date += timedelta(days=1)
                continue

            with self.archive.atomic_put_file(
                device_id=self.device_id,
                file_type='L2B_stratfinder',
                file_date=start_date,
                override=True,
            ) as output_file:
                ret |= qc_daily_final(
                    day_1a=yesterday_file,
                    day_2a=today_file,
                    day_3a=tomorrow_file,
                    config_filea=config_file,
                    file_values_qca=value_config_file,
                    config_attributes_file=stratfinder_metadata_file,
                    output_day2a=output_file,
                )

                if ret != 0:
                    print(f"Stratfinder QC failed for {start_date}, stopping")
                    print(f"Exit code: {ret}")
                    break

            start_date += timedelta(days=1)
        return ret

    def _validate_periods(self, periods: list[tuple[datetime, datetime]]) -> None:
        # check that start is before end
        for start, end in periods:
            if start >= end:
                raise ValueError(
                    f"Period start date {start} must be before end date {end}",
                )
        # check that periods do not overlap
        sorted_periods = sorted(periods, key=lambda x: x[0])
        for (start1, end1), (start2, end2) in zip(sorted_periods, sorted_periods[1:]):
            if end1 > start2:
                raise ValueError(
                    f"Periods must not overlap: {start1} - {end1} | {start2} - {end2}",
                )

    def derive_median_dark_measurement_profile(
            self,
            periods: list[tuple[datetime, datetime]] | datetime,
            prefix: str = 'live_',
            discard_window: timedelta = timedelta(minutes=20),
            calibration_window: timedelta = timedelta(minutes=30),
            window_tolerance: timedelta = timedelta(seconds=30),
            smooth_window: int = 10,
            smooth_above: float = 200,
    ) -> xr.Dataset:
        """Derive the median dark measurement profile from a hood measurement
        starting at ``start_date``. The first ``discard_window`` is discarded to
        let the sensor settle (Kotthaus et al. 2016), then ``calibration_window``
        of data is used to compute the median.

        The internally applied range and overlap corrections are reversed before
        taking the median, and for range gates above ``smooth_above`` a
        right-aligned running mean of width ``smooth_window`` is applied.

        Available components (``beta_att``, ``x_pol``, ``p_pol``) are written to
        ``self.calibration_profile_beta`` / ``_x_pol`` / ``_p_pol`` as
        ``DataArray`` s with ``float32`` range coordinate so they line up
        exactly with raw L1 ``range`` during the subtraction.

        :param periods: Can be a single start of a hood measurement, but if multiple
            were performed it can be a list of several periods represented as
            ``[(start, end), (start, end), ...]``. If a single datetime or string is
            provided, it will be treated as the start of a single hood measurement,
            and the end will be inferred as
            ``start + discard_window + calibration_window``. If multiple periods are
            provided, the median will be taken over the combined data from all periods.
            To allow for a very precise selection of  periods, ``discard_window`` and
            ``calibration_window`` are ignored when the inputs are provided as periods.
        :param prefix: Raw file name prefix.
        :param discard_window: Duration discarded at the start of the hood
            measurement to let the sensor settle.
        :param calibration_window: Duration of data used for the median.
        :param window_tolerance: Tolerance added to the observed time span when
            checking that it covers ``calibration_window`` (accounts for the
            granularity of timestamps in the raw files).
        :param smooth_window: Width of the right-aligned running mean (in range
            gates) applied above ``smooth_above``.
        :param smooth_above: Range (in m) above which smoothing is applied.
        :return: Dataset containing the smoothed median dark measurement
            profile for whichever of ``beta_att`` / ``x_pol`` / ``p_pol`` were
            present in the raw files.
        """
        prepped_ds = []
        auto_raning = False
        if isinstance(periods, datetime):
            auto_raning = True
            start = periods + discard_window
            end = start + calibration_window
            periods = [(start, end)]
        self._validate_periods(periods)
        for (_start, _end) in periods:
            files_needed = self.get_raw_files(_start, _end, prefix=prefix)
            print(
                f'getting {len(files_needed)} files for deriving the median dark '
                f'measurement profile',
            )
            with xr.open_mfdataset(
                paths=files_needed,
                combine='by_coords',
                data_vars='minimal',
                coords='minimal',
                compat='override',
            ).sel(time=slice(_start, _end)) as ds:
                # check what period we actually got and add some tolerance
                time_delta = (ds.time.max() - ds.time.min()).values + \
                    np.timedelta64(window_tolerance)
                if auto_raning and (time_delta < np.timedelta64(calibration_window)):
                    raise ValueError(
                        f"The calibration window of {calibration_window.seconds / 60:.1f} "  # noqa: E501
                        f"minutes could not be filled with data. The actual time range "
                        f"covered by the files is only "
                        f"{time_delta / np.timedelta64(1, 'm'):.1f} minutes. Consider "
                        f"reducing the discard_window or calibration_window, or check "
                        f"if the raw files are correctly stored in the "
                        f"input directory.",
                    )
                alc_vars = ('beta_att', 'x_pol', 'p_pol')
                # Reverse the overlap correction only if it was actually applied;
                # collapse the (time-invariant) function to a range-only profile so
                # it broadcasts cleanly both here and when it is re-applied below.
                overlap_reversed = (
                    'overlap_function' in ds and ds.overlap_is_corrected == 1
                )
                overlap_profile = None
                if overlap_reversed:
                    overlap_profile = ds['overlap_function'].fillna(1.0)
                for var in alc_vars:
                    if var in ds:
                        # 1. reverse the range correction
                        ds[var] = ds[var] / (ds['range'] ** 2)
                        # 2. reverse the overlap correction if it was applied
                        if overlap_reversed:
                            ds[var] = ds[var] * overlap_profile

                prepped_ds.append(ds.load())

        ds = xr.concat(prepped_ds, dim='time', data_vars='minimal')
        present = [var for var in alc_vars if var in ds]
        median_profiles = ds[present].median(dim='time')
        smoothed = median_profiles.rolling(range=smooth_window).mean()
        median_profiles = smoothed.where(
            median_profiles['range'] > smooth_above,
            median_profiles,
        )
        # range comes back as float64 after the arithmetic above; cast to
        # float32 so it aligns exactly with the float32 range coord on raw
        # L1 datasets during calibration subtraction.
        median_profiles = median_profiles.assign_coords(
            range=median_profiles['range'].astype('float32'),
        )

        self.calibration_profile_beta = (
            median_profiles['beta_att'] if 'beta_att' in median_profiles else None
        )
        self.calibration_profile_x_pol = (
            median_profiles['x_pol'] if 'x_pol' in median_profiles else None
        )
        self.calibration_profile_p_pol = (
            median_profiles['p_pol'] if 'p_pol' in median_profiles else None
        )
        return median_profiles

    def median_over_range_plot(
            self,
            output_path: str,
            profiles: xr.Dataset | None = None,
            alt_max: int | None = None,
            **kwargs: dict[str, Any],
    ) -> Figure:
        """Make a plot of the median dark measurement profile over range.

        :param output_path: The path to save the plot to.
        :param profiles: The dataset containing the profiles to plot. If ``None``,
            the profiles stored in the class instance (e.g. from a previous call to
            ``derive_median_dark_measurement_profile``) will be used. This allows
            plotting of any profile.
        :param alt_max: The maximum altitude to plot. If ``None``, the maximum altitude
            in the dataset will be used.
        :param kwargs: Additional keyword arguments to pass to the xarray plotting
            function. This can be used to customize the plot.
        :return: The figure object of the plot.
        """
        if profiles is not None:
            beta_att_profile = profiles.get('beta_att')
            x_pol_profile = profiles.get('x_pol')
            p_pol_profile = profiles.get('p_pol')
        else:
            beta_att_profile = self.calibration_profile_beta
            x_pol_profile = self.calibration_profile_x_pol
            p_pol_profile = self.calibration_profile_p_pol

        axs: list[Axes]
        if (
            beta_att_profile is not None
            and x_pol_profile is not None
            and p_pol_profile is not None
        ):
            fig, axs = plt.subplots(ncols=3, figsize=(12, 8), sharey=True)
        elif beta_att_profile is not None:
            fig, axs = plt.subplots(ncols=1, figsize=(4, 8), sharey=True)
            axs = [axs]
        else:
            raise ValueError(
                "The profile must contain at least 'beta_att' to be plotted. "
                "The presence of 'x_pol' and 'p_pol' is optional but if they are "
                "present, 'beta_att' must also be present.",
            )
        beta_att_profile.plot.line(
            y='range',
            ax=axs[0],
            label='beta_att',
            color='black',
            lw=0.4,
            **kwargs,
        )
        axs[0].set_xlabel(r'$\beta\;(m^{-1}\,sr^{-1})$')
        if x_pol_profile is not None:
            x_pol_profile.plot.line(
                y='range',
                ax=axs[1],
                label='x_pol',
                color='blue',
                lw=0.4,
                **kwargs,
            )
            axs[1].set_xlabel(r'$\beta_{xpol}\;(m^{-1}\,sr^{-1})$')
        if p_pol_profile is not None:
            p_pol_profile.plot.line(
                y='range',
                ax=axs[2],
                label='p_pol',
                color='green',
                lw=0.4,
                **kwargs,
            )
            axs[2].set_xlabel(r'$\beta_{ppol}\;(m^{-1}\,sr^{-1})$')
        for ax in axs:
            ax.set_title(None)
            ax.set_ylabel(None)
            ax.axvline(0, color='black', label='surface', zorder=0)
            ax.set_xlim(-1e-14, 1e-13)
            ax.set_axisbelow(True)
            ax.grid()
            if alt_max is not None:
                ax.set_ylim(0, alt_max)

        axs[0].set_ylabel('Range (m)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        return fig

    def beta_plot(
            self,
            start_date: datetime,
            end_date: datetime,
            output_path: str,
            alt_max: int | None = None,
            show_mlh: bool = False,
            show_ablh: bool = False,
            show_cbh: bool = False,
            filter_qc: bool = True,
            resampler: Callable[
                [xr.Dataset, timedelta],
                xr.Dataset,
            ] = resample_dataset,
            beta_file_type: Literal['L1', 'L2A_beta'] = 'L2A_beta',
            **kwargs: dict[str, Any],
    ) -> Figure:
        """Make a plot of the backscatter coefficient (beta) over time

        :param start_date: The start date of the plot.
        :param end_date: The end date of the plot.
        :param output_path: The path to save the plot to.
        :param alt_max: The maximum altitude to plot. If None, the maximum altitude
            in the dataset will be used.
        :param show_mlh: Whether to show the mixed layer height (MLH) on the plot.
        :param show_ablh: Whether to show the aerosol boundary layer height (ABLH)
            on the plot.
        :param show_cbh: Whether to show the cloud base height (CBH) on the plot.
        :param filter_qc: Whether to filter the MLH, ABLH, and CBH values based on
            the quality flag. If True, only values with a quality flag of 0 and a
            precipitation flag of 0 will be shown.
        :param resampler: A function that takes an xarray Dataset and a timedelta and
            returns a resampled xarray Dataset. This can be used to customize the
            resampling of the data, e.g. by using a different resampling method or by
            resampling to a different time resolution.
        :param beta_file_type: The file type to use for the beta data. This can be
            either 'L1' or 'L2A_beta'.
        :param kwargs: Additional keyword arguments to pass to the xarray plotting
            function. This can be used to customize the plot, e.g. by changing the
            colormap or the colorbar settings.
        :return: The figure object of the plot.
        """
        with self.archive.open_dataset(
            device_id=self.device_id,
            file_type=beta_file_type,
            start_date=start_date,
            end_date=end_date,
            engine='netcdf4',
            data_vars='minimal',
            compat='override',
            coords='minimal',
        ) as ds:
            delta = end_date - start_date
            # capture station coordinates before subsetting/resampling since
            # minimal/data_vars settings or selecting variables may drop scalar
            # metadata variables like station_latitude/station_longitude
            try:
                # this is for L1
                lat = float(ds['station_latitude'].item())
                lon = float(ds['station_longitude'].item())
            except NotImplementedError:
                # this is for L2A_beta
                lat = float(ds['station_latitude'].values[0])
                lon = float(ds['station_longitude'].values[0])

            ds = resampler(ds[['beta']], delta)
            # compute log10 of beta
            ds['log10_beta'] = np.log10(ds.beta.where(ds.beta > 0))
            fig, ax = plt.subplots(figsize=(12, 7))

            ds.log10_beta.plot(
                x='time',
                vmin=-7,
                vmax=-4,
                cmap='turbo',
                cbar_kwargs={
                    'label': r'$log_{10}(\beta)\ (m^{-1}\ sr^{-1})$',
                    'location': 'bottom',
                    'shrink': 0.5,
                    'pad': 0.15,
                },
                ax=ax,
                **kwargs,
            )
            add_solar_times(ax, ds, lat=lat, lon=lon)

        if any([show_mlh, show_ablh, show_cbh]):
            with self.archive.open_dataset(
                device_id=self.device_id,
                file_type='L2B_stratfinder',
                start_date=start_date,
                end_date=end_date,
                engine='netcdf4',
                data_vars='minimal',
                compat='override',
                coords='minimal',
            ) as _ds_strat:
                ds_strat = resampler(_ds_strat, delta)
                if filter_qc:
                    # let's filter out low-quality points
                    ds_strat = ds_strat.where(
                        (ds_strat.quality_FLAG == 0) & (
                            ds_strat.precip_FLAG == 0
                        ),
                    )
                if show_ablh:
                    ds_strat['ABLH'].plot.line(
                        x='time',
                        ax=ax,
                        label='ABLH',
                        color='white',
                        path_effects=[
                            mpe.Stroke(linewidth=2.25, foreground='grey'),
                            mpe.Stroke(foreground='white', alpha=1),
                            mpe.Normal(),
                        ],
                        lw=1,
                    )
                if show_mlh:
                    ds_strat['MLH'].plot(
                        x='time',
                        ax=ax,
                        label='MLH',
                        color='white',
                        path_effects=[
                            mpe.Stroke(linewidth=2.25, foreground='red'),
                            mpe.Stroke(foreground='white', alpha=1),
                            mpe.Normal(),
                        ],
                        lw=0.75,
                    )
                if show_cbh:
                    ds_strat['cloud_base_altitude'].plot.scatter(
                        x='time',
                        y='altitude',
                        ax=ax,
                        label='CBH',
                        color='white',
                        edgecolor='black',
                        marker='o',
                        linewidth=0.5,
                        s=20,
                    )

        ax.set_title(None)
        ax.set_ylabel('altitude (m agl)')
        ax.set_xlabel('time (UTC)')
        ax.legend(loc='upper right')
        if alt_max is not None:
            ax.set_ylim(0, alt_max)

        ax.grid()
        fig.autofmt_xdate()
        ax.set_xlim(start_date, end_date)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        return fig

    def ldr_plot(
            self,
            start_date: datetime,
            end_date: datetime,
            output_path: str,
            alt_max: int | None = None,
            resampler: Callable[
                [xr.Dataset, timedelta],
                xr.Dataset,
            ] = resample_dataset,
            **kwargs: dict[str, Any],
    ) -> Figure:
        """Make a plot of the linear depolarisation ratio (LDR) over time

        :param start_date: The start date of the plot.
        :param end_date: The end date of the plot.
        :param output_path: The path to save the plot to.
        :param alt_max: The maximum altitude to plot. If None, the maximum altitude
            in the dataset will be used.
        :param kwargs: Additional keyword arguments to pass to the xarray plotting
            function. This can be used to customize the plot, e.g. by changing the
            colormap or the colorbar settings.
        :return: The figure object of the plot.
        """
        with self.archive.open_dataset(
            device_id=self.device_id,
            file_type='L1',
            start_date=start_date,
            end_date=end_date,
            engine='netcdf4',
            data_vars='minimal',
            compat='override',
            coords='minimal',
        ) as ds:
            delta = end_date - start_date
            # capture coordinates before selecting variables
            lat = float(ds['station_latitude'].item())
            lon = float(ds['station_longitude'].item())
            if 'linear_depol_ratio' not in ds:
                raise KeyError(
                    'The linear_depol_ratio variable is not available in the dataset. '
                    'This variable is only available in L1 files if the ceilometer '
                    'supports measuring it. Please check if your device supports this'
                    'variable.',
                )
            ds = resampler(ds[['linear_depol_ratio']], delta)

            ds = ds.linear_depol_ratio.where(ds.linear_depol_ratio < 0.69).where(
                ds.linear_depol_ratio > 0.001,
            )
            fig, ax = plt.subplots(figsize=(12, 7))
            ds.plot(
                x='time',
                vmin=0,
                vmax=0.7,
                cbar_kwargs={
                    'label': 'Linear Depolarisation ratio (-)',
                    'location': 'bottom',
                    'shrink': 0.5,
                },
                cmap=LDR_CMAP,
                ax=ax,
                **kwargs,
            )
            add_solar_times(ax, ds, lat=lat, lon=lon)

        ax.set_ylabel('altitude (m agl)')
        ax.set_xlabel('time (UTC)')
        if alt_max is not None:
            ax.set_ylim(0, alt_max)

        ax.set_title(None)
        ax.grid()
        fig.autofmt_xdate()
        ax.set_xlim(start_date, end_date)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        return fig
