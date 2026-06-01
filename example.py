from datetime import datetime

import xarray as xr
from ceilometer_toolbox import Ceilometer
from ceilometer_toolbox import CeilometerArchive


def main() -> int:
    archive = CeilometerArchive('ceilometer-output')
    profiles = xr.open_dataset('example_configs/median_dark_measurement_profile.nc')
    ceilometer = Ceilometer(
        device_id='IA',
        input_dir='ceilometer-input',
        archive=archive,
        raw2l1_config_file='example_configs/raw2l1_cl61.conf',
        stratfinder_config_file='example_configs/stratfinder_settings_cl61.json',
        stratfinder_qc_value_config_file='example_configs/values_qc.toml',
        stratfinder_qc_metadata_file='example_configs/STRATFINDER_metadata.toml',
        calibration_profile_beta=profiles['beta_att'],
        calibration_profile_p_pol=profiles['p_pol'],
        calibration_profile_x_pol=profiles['x_pol'],
    )
    ceilometer.process_raw_files(
        start_date='2026-04-27', end_date='2026-04-27', jobs=4,
    )
    ceilometer.process_l1_files(start_date='2026-04-27', end_date='2026-04-28')
    ceilometer.process_stratfinder_qc(
        start_date='2026-04-27', end_date='2026-04-28',
    )
    ceilometer.beta_plot(
        start_date=datetime(2026, 4, 28),
        end_date=datetime(2026, 5, 2),
        show_mlh=True,
        show_ablh=True,
        show_cbh=True,
        alt_max=2500,
        output_path='beta_plot.png',
    )
    ceilometer.ldr_plot(
        start_date=datetime(2026, 4, 28),
        end_date=datetime(2026, 5, 2),
        alt_max=2500,
        output_path='ldr_plot.png',
    )
    ceilometer.median_over_range_plot(
        output_path='median_dark_measurement_profile.png',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
