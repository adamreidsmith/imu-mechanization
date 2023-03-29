import csv
import time
from math import pi
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mechanization import INSMechanization
from mechanization_pure import INSMechanization as PureINSMechanization


def main():
    # Read the data
    data = np.fromfile('./project_data.BIN').reshape([-1, 7])

    # IMU parameters, converted to SI base units
    gyro_bias = 0.1 * pi / 180 / 3600  # rad / s
    arw = 0.01 * pi / 180 / 60  # rad / sqrt(s)
    gyro_bias_instability = 0.015 * pi / 180 / 3600  # rad / s
    gyro_corr_time = 3600  # s
    # This is a function as it depends on g
    acc_bias = lambda g: 3 * abs(g) * 1e-6  # m / s ** 2
    vrw = 0.003 / 60  # m / s ** (3/2)
    # This is a function as it depends on g
    acc_bias_instability = lambda g: 50 * abs(g) * 1e-6  # m / s ** 2
    acc_corr_time = 3600  # s

    # Set the scale factor and non-orthogoanlity matrices
    # scale_factor = np.eye(3)
    scale_factor = np.zeros((3, 3))
    nonorthogonality = np.zeros((3, 3))

    # Initial position (converted to radians) and velocity
    lat = 51.07995352 * pi / 180  # rad
    long = -114.13371127 * pi / 180  # rad
    h = 1118.502  # m

    # Instantiate the model
    INS = INSMechanization(
        h,
        lat,
        long,
        acc_bias,
        gyro_bias,
        accel_sf=scale_factor,
        gyro_sf=scale_factor,
        accel_no=nonorthogonality,
        gyro_no=nonorthogonality,
        alignment_time=300,
    )

    # Instantiate the model written in pure python
    INSPure = PureINSMechanization(
        h,
        lat,
        long,
        acc_bias,
        gyro_bias,
        accel_sf=scale_factor,
        gyro_sf=scale_factor,
        accel_no=nonorthogonality,
        gyro_no=nonorthogonality,
        alignment_time=300,
    )

    # # Compute results
    # results = []
    # t0 = time.perf_counter()
    # for i, measurement in enumerate(data):
    #     INS.process_measurement(measurement)
    #     results.append(INS.get_params())
    # print(f'Mechanization completed in {time.perf_counter() - t0:.3f} seconds')

    data = data.tolist()

    results = []
    t0 = time.perf_counter()
    for i, measurement in enumerate(data):
        INSPure.process_measurement(measurement)
        results.append(INSPure.get_params())
    print(f'Pure mechanization completed in {time.perf_counter() - t0:.3f} seconds')

    # Save the results in csv format
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(INS.get_params(get_labels=True))
        writer.writerows(results)

    results = np.array([r[:-1] for r in results if not r[-1]])

    # Plot the results
    timestamps = results[:, 0] / 60
    results = results[:, 1:]
    units = (
        'degrees',
        'degrees',
        'meters',
        'm/s',
        'm/s',
        'm/s',
        'degrees',
        'degrees',
        'degrees',
    )
    labels = INS.get_params(get_labels=True)
    plt.figure(figsize=(15, 10.5))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        sns.lineplot(x=timestamps, y=results[:, i], label=labels[i + 1])
        plt.xlabel('time (min)')
        plt.ylabel(units[i])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
