import os
import subprocess
import tempfile
from collections.abc import Callable
from datetime import date
from datetime import datetime
from datetime import timedelta
from glob import glob
from multiprocessing import Pool
from typing import Any

import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ceilometer_toolbox.data import atomic_write_path
from ceilometer_toolbox.data import CeilometerArchive
from ceilometer_toolbox.utils import add_solar_times
from ceilometer_toolbox.utils import LDR_CMAP
from ceilometer_toolbox.utils import resample_dataset
from matplotlib.figure import Figure
from qc_sf_python.qc_daily_final import qc_daily_final
from raw2l1.raw2l1 import raw2l1


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
        """
        self.archive = archive
        self.input_dir = input_dir
        self.device_id = device_id
        self.raw2l1_config_file = raw2l1_config_file
        self.stratfinder_config_file = stratfinder_config_file
        self.stratfinder_qc_value_config_file = stratfinder_qc_value_config_file
        self.stratfinder_qc_metadata_file = stratfinder_qc_metadata_file

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

        :return: The return code of the raw2l1 tool, 0 if successful, non-zero otherwise
        """
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
            if isinstance(
                input_files,
                str,
            )
            else input_files
        )
        ancillary_files = (
            [ancillary_files]
            if isinstance(
                ancillary_files,
                str,
            )
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
            return ret

    def process_raw_files(
            self,
            start_date: date | str | None = None,
            end_date: date | str | None = None,
            prefix: str = 'live_',
            jobs: int = 1,
            config_file: str | None = None,
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
        """
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
            container_image: str = 'stratfinder:latest',
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

        # now modify the paths to match the docker container's file system, which
        # is just the current working directory
        today_file = os.path.join('/data', today_file)
        output_file = os.path.join('/data', os.path.relpath(output_file))
        beta_file = os.path.join('/data', os.path.relpath(beta_file))
        yesterday_file = (
            os.path.join(
                '/data',
                yesterday_file,
            )
            if yesterday_file
            else None
        )
        overlap_file = (
            os.path.join(
                '/data',
                overlap_file,
            )
            if overlap_file
            else None
        )

        # also adjust this path for the docker container
        dyn_config = os.path.join('/data', config_file)

        cmd = (
            'docker',
            'run',
            '-e',
            'AGREE_TO_MATLAB_RUNTIME_LICENSE=yes',
            # use the current user to get permissions right in the folder
            '-u',
            f"{os.getuid()}:{os.getgid()}",
            '--rm',
            '-ti',
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
            start_date: date | str,
            end_date: date | str | None = None,
            config_file: str | None = None,
            directory_mount: str | None = None,
            in_docker: bool = True,
            executable_path: str | None = None,
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
        """
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
            start_date: date | str,
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
            if not all([yesterday_file, today_file]):
                print(
                    f"One of the required files for {start_date} does not exist",
                )
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
            # capture station coordinates before subsetting/resampling since
            # minimal/data_vars settings or selecting variables may drop scalar
            # metadata variables like station_latitude/station_longitude
            lat = float(ds['station_latitude'].item())
            lon = float(ds['station_longitude'].item())

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
            )
            add_solar_times(ax, ds, lat=lat, lon=lon)

        ax.set_ylabel('altitude (m agl)')
        ax.set_xlabel('time (UTC)')
        if alt_max is not None:
            ax.set_ylim(0, alt_max)

        ax.grid()
        fig.autofmt_xdate()
        ax.set_xlim(start_date, end_date)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        return fig
