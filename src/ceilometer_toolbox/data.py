import os
import re
import tempfile
from collections.abc import Generator
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import cast
from typing import Literal
from typing import NamedTuple

import xarray as xr

# L1: raw2l1 output
# L2A_beta: stratfinder beta output
# L2A_stratfinder: stratfinder output
# L2B_stratfinder: stratfinder output after qc application
FileType = Literal['L1', 'L2A_beta', 'L2A_stratfinder', 'L2B_stratfinder']


class ArchiveFile(NamedTuple):
    '''One discovered ceilometer file in the archive tree.'''

    device_id: str
    file_type: FileType
    file_date: date
    full_path: str


@contextmanager
def atomic_write_path(
        final_path: str,
        override: bool = False,
) -> Generator[str]:
    '''Prepare staged output path and atomically publish to final path.

    This context manager yields a temporary file path located in a unique
    temporary directory next to ``final_path``. Downstream tools can write to
    this path.
    On successful context exit, the staged file is moved into place using
    ``os.replace``.

    :param final_path: target path for final published file
    :param override: allow replacing existing target file when ``True``
    :return: iterator yielding temporary staged output path
    :raises FileExistsError: if target exists and ``override`` is ``False``
    '''
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    if os.path.exists(final_path) and not override:
        raise FileExistsError(f'File already exists: {final_path}')

    with tempfile.NamedTemporaryFile(
        prefix=f'.{os.path.basename(final_path)}.',
        dir=os.path.dirname(final_path),
    ) as tmp_path:
        yield tmp_path.name
        if os.path.exists(tmp_path.name):
            if os.path.exists(final_path) and not override:
                raise FileExistsError(f'File already exists: {final_path}')
            # change file permission to 0o644 to be readable by other users, which
            # is the usual umask
            os.chmod(tmp_path.name, 0o644)
            os.replace(tmp_path.name, final_path)


class CeilometerArchive:
    '''Query ceilometer output files stored as daily NetCDF files.

    Expected directory layout:
    ``<root>/<device_id>/<YYYY>/<MM>/<YYYYMMDD>_<file_type>.nc``
    '''

    VALID_FILE_TYPES: frozenset[FileType] = frozenset(
        # TODO: research how IPSL does that
        ('L1', 'L2A_beta', 'L2A_stratfinder', 'L2B_stratfinder'),
    )
    _FILE_PATTERN = re.compile(
        fr'^(\d{{8}})_({"|".join(VALID_FILE_TYPES)})\.nc$',
    )

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        if not os.path.isdir(root_dir):
            raise ValueError(f'Root directory does not exist: {root_dir}')

    @staticmethod
    def _parse_date(value: str | date | datetime) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            for fmt in ('%Y-%m-%d', '%Y%m%d'):
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    continue
        raise ValueError(
            'Date values must be date, datetime, or strings in '
            'YYYY-MM-DD/YYYYMMDD format.',
        )

    @staticmethod
    def _iter_days(start_date: date, end_date: date) -> Iterator[date]:
        current = start_date
        while current <= end_date:
            yield current
            current += timedelta(days=1)

    def _iter_device_ids(self, device_id: str | None = None) -> Iterator[str]:
        '''Yield matching device IDs from the archive root.

        If ``device_id`` is provided and no corresponding directory exists,
        this iterator yields nothing.
        '''
        if device_id is not None:
            device_dir = os.path.join(self.root_dir, device_id)
            if os.path.isdir(device_dir):
                yield device_id
            return

        if not os.path.isdir(self.root_dir):
            return
        for found_device in sorted(os.listdir(self.root_dir)):
            if os.path.isdir(os.path.join(self.root_dir, found_device)):
                yield found_device

    def _available_files(
            self,
            device_id: str | None = None,
            file_type: FileType | None = None,
            start_date: date | None = None,
            end_date: date | None = None,
    ) -> list[ArchiveFile]:
        '''Return matching files from the archive tree.

        :param device_id: optional station/device ID filter
        :param file_type: optional file type filter
        :param start_date: optional inclusive lower bound
        :param end_date: optional inclusive upper bound
        :return: ArchiveFile entries with device_id, file_type, file_date,
            and full_path
        '''
        if file_type is not None:
            self._validate_file_type(file_type)

        if start_date is not None and end_date is not None and start_date > end_date:  # noqa: E501
            return []

        devices = list(self._iter_device_ids(device_id=device_id))
        if not devices:
            return []

        selected_types: list[FileType] = [file_type] if file_type is not None else sorted(  # noqa: E501
            self.VALID_FILE_TYPES,
        )
        rows: list[ArchiveFile] = []
        # fully bounded date range -> check only concrete candidate files.
        if start_date is not None and end_date is not None:
            for found_device in devices:
                for day in self._iter_days(start_date, end_date):
                    for found_type in selected_types:
                        path = os.path.join(
                            self.root_dir,
                            found_device,
                            f'{day:%Y}',
                            f'{day:%m}',
                            f'{day:%Y%m%d}_{found_type}.nc',
                        )
                        if os.path.isfile(path):
                            rows.append(
                                ArchiveFile(
                                    device_id=found_device,
                                    file_type=found_type,
                                    file_date=day,
                                    full_path=path,
                                ),
                            )
            return rows

        # Partial range or unbounded -> list only candidate month folders.
        for found_device in devices:
            device_dir = os.path.join(self.root_dir, found_device)
            if not os.path.isdir(device_dir):
                continue

            year_dirs = [
                d for d in sorted(os.listdir(device_dir))
                if d.isdigit() and os.path.isdir(os.path.join(device_dir, d))
            ]
            for year_dir_name in year_dirs:
                year_int = int(year_dir_name)
                if start_date is not None and year_int < start_date.year:
                    continue
                if end_date is not None and year_int > end_date.year:
                    continue

                year_dir = os.path.join(device_dir, year_dir_name)
                month_dirs = [
                    d for d in sorted(os.listdir(year_dir))
                    if d.isdigit() and os.path.isdir(os.path.join(year_dir, d))
                ]
                for month_dir_name in month_dirs:
                    month_int = int(month_dir_name)
                    if (
                        start_date is not None
                        and (year_int, month_int) < (start_date.year, start_date.month)  # noqa: E501
                    ):
                        continue
                    if (
                        end_date is not None
                        and (year_int, month_int) > (end_date.year, end_date.month)  # noqa: E501
                    ):
                        continue

                    month_dir = os.path.join(year_dir, month_dir_name)
                    for filename in sorted(os.listdir(month_dir)):
                        match = self._FILE_PATTERN.match(filename)
                        if match is None:
                            continue

                        found_type_candidate = match.group(2)
                        if found_type_candidate not in selected_types:
                            continue
                        found_type = cast(FileType, found_type_candidate)

                        found_date = datetime.strptime(
                            match.group(1), '%Y%m%d',
                        ).date()
                        if start_date is not None and found_date < start_date:
                            continue
                        if end_date is not None and found_date > end_date:
                            continue

                        rows.append(
                            ArchiveFile(
                                device_id=found_device,
                                file_type=found_type,
                                file_date=found_date,
                                full_path=os.path.join(month_dir, filename),
                            ),
                        )
        return rows

    def _validate_file_type(self, file_type: FileType | str) -> None:
        '''Validate that ``file_type`` is supported by the archive.'''
        if file_type not in self.VALID_FILE_TYPES:
            valid = ', '.join(sorted(self.VALID_FILE_TYPES))
            raise ValueError(
                f'Unsupported file type: {file_type}. Use one of: {valid}',
            )

    def _file_path(
            self,
            device_id: str,
            file_type: FileType,
            file_date: str | date | datetime,
    ) -> str:
        '''Build the canonical archive path for one device, type and date.

        :param device_id: station/device ID
        :param file_type: one of ``L1``, ``L2A_beta``, ``L2A_stratfinder``
            or ``L2B_stratfinder``
        :param file_date: target file date
        :return: full canonical path (file need not exist)
        :raises ValueError: if ``file_type`` is unsupported
        '''
        self._validate_file_type(file_type)
        parsed_date = self._parse_date(file_date)
        return os.path.join(
            self.root_dir,
            device_id,
            f'{parsed_date:%Y}',
            f'{parsed_date:%m}',
            f'{parsed_date:%Y%m%d}_{file_type}.nc',
        )

    def latest_date(
            self,
            device_id: str,
            file_type: FileType,
            from_date: str | date | datetime | None = None,
            max_depth_days: int = 3660,
    ) -> date | None:
        '''Return the newest available date with stop-early backward traversal.

        :param device_id: station/device ID
        :param file_type: one of ``L1``, ``L2A_beta``, ``L2A_stratfinder``
            or ``L2B_stratfinder``
        :param from_date: optional start point for backward search
            (defaults to today)
        :param max_depth_days: maximum number of days to look back (inclusive)
        :return: latest available date, or ``None`` if no file exists
        :raises ValueError: if ``max_depth_days`` is negative
        '''
        self._validate_file_type(file_type)
        if max_depth_days < 0:
            raise ValueError('max_depth_days must be >= 0')

        anchor = date.today() if from_date is None else self._parse_date(from_date)  # noqa: E501
        for offset in range(max_depth_days + 1):
            candidate = date.fromordinal(anchor.toordinal() - offset)
            if os.path.isfile(self._file_path(device_id, file_type, candidate)):  # noqa: E501
                return candidate
        return None

    def put_file(
            self,
            device_id: str,
            file_type: FileType,
            file_date: str | date | datetime,
            override: bool = False,
    ) -> str:
        '''Prepare one archive file path and create missing directories.

        This method only resolves and prepares the path for downstream tools.
        It does not write file contents.

        :param device_id: station/device ID
        :param file_type: one of ``L1``, ``L2A_beta``, ``L2A_stratfinder``
            or ``L2B_stratfinder``
        :param file_date: target file date
        :param override: allow existing file path when ``True``
        :return: prepared full path
        :raises FileExistsError: if target exists and ``override`` is ``False``
        :raises ValueError: if ``file_type`` is unsupported
        '''
        path = self._file_path(device_id, file_type, file_date)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not override:
            raise FileExistsError(f'File already exists: {path}')
        return path

    @contextmanager
    def atomic_put_file(
            self,
            device_id: str,
            file_type: FileType,
            file_date: str | date | datetime,
            override: bool = False,
    ) -> Generator[str]:
        '''Prepare one archive output path for atomic publication.

        This is the atomic variant of :meth:`put_file`. It yields a temporary
        path in the target directory and atomically publishes to the canonical
        archive path on successful context exit.

        :param device_id: station/device ID
        :param file_type: one of ``L1``, ``L2A_beta``, ``L2A_stratfinder``
            or ``L2B_stratfinder``
        :param file_date: target file date
        :param override: allow replacing existing target file when ``True``
        :return: iterator yielding temporary staged output path
        :raises FileExistsError: if target exists and ``override`` is ``False``
        :raises ValueError: if file type is unsupported
        '''
        final_path = self._file_path(device_id, file_type, file_date)
        with atomic_write_path(final_path, override=override) as tmp:
            yield tmp

    def delete_file(
            self,
            device_id: str,
            file_type: FileType,
            file_date: str | date | datetime,
    ) -> bool:
        '''Delete one archive file from the tree.

        :param device_id: station/device ID
        :param file_type: one of ``L1``, ``L2A_beta``, ``L2A_stratfinder``
            or ``L2B_stratfinder``
        :param file_date: file date to delete
        :return: ``True`` if a file was deleted, otherwise ``False``
        '''
        path = self._file_path(device_id, file_type, file_date)
        if not os.path.isfile(path):
            return False
        os.remove(path)
        return True

    def iter_files(
            self,
            device_id: str | None = None,
            file_type: FileType | None = None,
            start_date: str | date | datetime | None = None,
            end_date: str | date | datetime | None = None,
    ) -> Iterator[str]:
        '''Yield file paths for file type and an inclusive date interval.

        Defaults:
        - device_id=None -> all available devices
        - file_type=None -> all supported file types
        - start_date/end_date=None -> min/max available dates in the archive

        :param device_id: optional station/device ID filter
        :param file_type: optional file type filter
            (``L1``, ``L2A_beta``, ``L2A_stratfinder``, ``L2B_stratfinder``)
        :param start_date: optional inclusive lower bound
        :param end_date: optional inclusive upper bound
        :return: iterator over matching file paths
        :raises ValueError: if file type is unsupported or
            start_date > end_date
        '''
        explicit_start = self._parse_date(
            start_date,
        ) if start_date is not None else None
        explicit_end = self._parse_date(
            end_date,
        ) if end_date is not None else None
        if explicit_start is not None and explicit_end is not None and explicit_start > explicit_end:  # noqa: E501
            raise ValueError('start_date must be <= end_date')

        available = self._available_files(
            device_id=device_id,
            file_type=file_type,
            start_date=explicit_start,
            end_date=explicit_end,
        )
        for row in available:
            yield row.full_path

    def get_files(
            self,
            device_id: str | None = None,
            file_type: FileType | None = None,
            start_date: str | date | datetime | None = None,
            end_date: str | date | datetime | None = None,
    ) -> list[str]:
        '''Return list variant of :meth:`iter_files` for convenience.

        Accepts the same arguments as :meth:`iter_files` and raises the same
        exceptions.
        '''
        return list(
            self.iter_files(
                device_id=device_id,
                file_type=file_type,
                start_date=start_date,
                end_date=end_date,
            ),
        )

    def get_file_or_none(
            self,
            device_id: str,
            file_type: FileType,
            file_date: str | date | datetime,
    ) -> str | None:
        '''Return the single file path for one device, type and date.

        :param device_id: station/device ID
        :param file_type: one of ``L1``, ``L2A_beta``, ``L2A_stratfinder``
            or ``L2B_stratfinder``
        :param file_date: target file date
        :return: full path to the matching file, or ``None`` if missing
        '''
        path = self._file_path(device_id, file_type, file_date)
        if not os.path.isfile(path):
            return None
        return path

    def _open_ds(
            self,
            device_id: str,
            file_type: FileType,
            start_date: str | date | datetime | None = None,
            end_date: str | date | datetime | None = None,
            **kwargs: Any,
    ) -> xr.Dataset:
        files = self.get_files(
            device_id=device_id,
            file_type=file_type,
            start_date=start_date,
            end_date=end_date,
        )
        if not files:
            raise FileNotFoundError(
                'No files found for '
                f'device_id={device_id}, file_type={file_type}, '
                f'start_date={start_date}, end_date={end_date}',
            )

        # ValueError: Resulting object does not have monotonic global indexes
        # along dimension altitude
        if file_type == 'L2A_beta':
            # fix the corrected altitude so we are even able to combine them!
            # this is somewhat of a hack - I am not sure why the resulting
            # altitudes are different
            with xr.open_dataset(files[0]) as ds:
                ref_alt = ds.altitude.copy()

            def _preprocess(ds: xr.Dataset) -> xr.Dataset:
                """Preprocess each dataset to reindex altitude to the reference

                This is needed since stratfinder returns slightly different
                altitudes for each file, which causes a mismatch in the
                combined dataset. We allow a tolerance of 2 meters, while
                expecting a level every ~5 m.
                """
                return ds.reindex(
                    altitude=ref_alt,
                    method='nearest',
                    tolerance=2.0,
                )

            dataset = xr.open_mfdataset(
                files,
                join='outer',
                preprocess=_preprocess,
                **kwargs,
            )
        else:
            dataset = xr.open_mfdataset(files, **kwargs)

        if start_date is None and end_date is None:
            return dataset

        slice_start = self._parse_date(
            start_date,
        ).isoformat() if start_date is not None else None
        slice_end = self._parse_date(
            end_date,
        ).isoformat() if end_date is not None else None
        if slice_start is not None or slice_end is not None:  # pragma: no branch
            dataset = dataset.sel(time=slice(slice_start, slice_end))
        return dataset

    @contextmanager
    def open_dataset(
            self,
            device_id: str,
            file_type: FileType,
            start_date: str | date | datetime | None = None,
            end_date: str | date | datetime | None = None,
            **kwargs: Any,
    ) -> Generator[xr.Dataset]:
        '''Open matching files as one xarray dataset and slice by date range.

        :param device_id: station/device ID
        :param file_type: one of ``L1``, ``L2A_beta``, ``L2A_stratfinder``
            or ``L2B_stratfinder``
        :param start_date: optional inclusive lower bound
        :param end_date: optional inclusive upper bound
        :param kwargs: additional keyword arguments passed to
            ``xarray.open_mfdataset``
        :return: xarray dataset containing the selected time range
        :raises FileNotFoundError: if no matching files are found
        :raises ValueError: if file type is unsupported or
            start_date > end_date
        '''
        ds = self._open_ds(
            device_id=device_id,
            file_type=file_type,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
        try:
            yield ds
        finally:
            ds.close()
