from math import sqrt, sin, cos, tan, asin, atan, pi
import numpy as np
from typing import Callable

RAD_TO_DEG = 180 / pi


class INSMechanization:
    '''
    Class to compute mechanization of IMU data.
    All values must be in SI base units or SI derived units.
    '''

    e_squared = 6.69438e-3  # squared WGS84 eccentricity of the Earth
    semimajor_axis = 6378137  # WGS84 semi-major axis of the Earth (m)
    omega_e = 7.292115147e-5  # angular velocity of the Earth (rad / s)

    def __init__(
        self,
        h0: float,  # initial height
        lat0: float,  # initial latitude
        long0: float,  # initial longitude
        accel_bias: float | Callable = 0.0,  # accel bias
        gyro_bias: float = 0.0,  # gyro bias
        accel_sf: float | np.ndarray = 0.0,  # accel scale factor matrix
        gyro_sf: float | np.ndarray = 0.0,  # gyro scale factor matrix
        accel_no: np.ndarray = np.zeros((3, 3)),  # accel non-orthogonality
        gyro_no: np.ndarray = np.zeros((3, 3)),  # gyro non-orthogonality
        alignment_time: float = 120,  # alignment time in seconds
    ) -> None:
        self.h: float = h0
        self.lat: float = lat0
        self.long: float = long0
        self.accel_bias: float | Callable = accel_bias
        self.gyro_bias: float = gyro_bias
        accel_sf: float | int | np.ndarray = accel_sf * np.eye(3) if isinstance(accel_sf, (float, int)) else accel_sf
        gyro_sf: float | int | np.ndarray = gyro_sf * np.eye(3) if isinstance(gyro_sf, (float, int)) else gyro_sf
        self.accel_inv_error_matrix: np.ndarray = np.linalg.inv(np.eye(3) + accel_sf + accel_no)
        self.gyro_inv_error_matrix: np.ndarray = np.linalg.inv(np.eye(3) + gyro_sf + gyro_no)
        self.quat: np.ndarray | None = None  # rotation quaternion; inititalized when alignment completes
        self.R_b2l: np.ndarray | None = None  # rotation matrix; inititalized when alignment completes
        self.v_llf: np.ndarray = np.zeros((3, 1))  # initial ENU velocities
        self.prev_delta_v_llf: np.ndarray = np.zeros((3, 1))  # change in v in LLF at previous t
        self.prev_time: float = 0  # previous time, used to determine delta_t
        self.roll: float = 0
        self.pitch: float = 0
        self.azimuth: float = 0
        self.timestamp: float = 0  # store the current timestamp
        self.start_time: float | None = None  # start time of the mechanization; initialized on the first iteration

        # Alignment specific attributes
        self.alignment_time: float | int = alignment_time
        self.alignment_complete: bool = False  # flag to determine if alignment is completed
        self.alignment_acc_mean: np.ndarray = np.zeros((3, 1))  # running mean for accel alignment
        self.alignment_omega_mean: np.ndarray = np.zeros((3, 1))  # running mean for gyro alignment
        self.alignment_it: int = 0  # couter to keep track of alignment iterations

        # Values for error tracking
        self.initial_attitude: np.ndarray | None = None  # initial roll, pitch, azimuth as computed by alignment
        self.position_errors: np.ndarray = np.zeros((3, 1))  # errors in position in the initial LLF

    @classmethod
    def get_rotation_matrix(cls, r: float, p: float, A: float) -> np.ndarray:
        '''Compute the rotation matrix to rotate the body frame to the LLF from the Euler angles'''

        return cls.Rz(-A) @ cls.Rx(p) @ cls.Ry(r)

    @staticmethod
    def matrix_to_normalized_quaternion(R: np.ndarray) -> np.ndarray:
        '''Convert a rotation matrix to a normalized quaternion'''

        q4 = sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        four_q4 = 4 * q4
        q1 = (R[2, 1] - R[1, 2]) / four_q4
        q2 = (R[0, 2] - R[2, 0]) / four_q4
        q3 = (R[1, 0] - R[0, 1]) / four_q4
        q_norm = sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
        return np.array([[q1 / q_norm], [q2 / q_norm], [q3 / q_norm], [q4 / q_norm]])

    @staticmethod
    def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
        '''Convert a normalized quaternion to a rotation matrix'''

        q = q.reshape(4)
        q1, q2, q3, q4 = q
        qq1, qq2, qq3, qq4 = q * q
        q1q2 = q1 * q2
        q3q4 = q3 * q4
        q1q3 = q1 * q3
        q2q4 = q2 * q4
        q1q4 = q1 * q4
        q2q3 = q2 * q3
        return np.array(
            [
                [qq1 - qq2 - qq3 + qq4, 2 * (q1q2 - q3q4), 2 * (q1q3 + q2q4)],
                [2 * (q1q2 + q3q4), qq2 + qq4 - qq1 - qq3, 2 * (q2q3 - q1q4)],
                [2 * (q1q3 - q2q4), 2 * (q2q3 + q1q4), qq3 + qq4 - qq1 - qq2],
            ]
        )

    @staticmethod
    def vec_to_skew(v: np.ndarray) -> np.ndarray:
        '''Convert a vector to its skew-symmetric representation'''

        v1, v2, v3 = v.reshape(3)
        return np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])

    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        '''Returns the rotation matrix about the x-axis by angle theta using the sign convention in Wikipedia'''

        costheta = cos(theta)
        sintheta = sin(theta)
        return np.array([[1, 0, 0], [0, costheta, -sintheta], [0, sintheta, costheta]])

    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        '''Returns the rotation matrix about the y-axis by angle theta using the sign convention in Wikipedia'''

        costheta = cos(theta)
        sintheta = sin(theta)
        return np.array([[costheta, 0, sintheta], [0, 1, 0], [-sintheta, 0, costheta]])

    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        '''Returns the rotation matrix about the z-axis by angle theta using the sign convention in Wikipedia'''

        costheta = cos(theta)
        sintheta = sin(theta)
        return np.array([[costheta, -sintheta, 0], [sintheta, costheta, 0], [0, 0, 1]])

    @staticmethod
    def gravity(lat: float, h: float) -> float:
        '''Compute the gravity vector given the latitude and height'''

        A = (
            9.7803267715,
            0.0052790414,
            2.32718e-5,
            -3.087691089e-6,
            4.397731e-9,
            7.21e-13,
        )
        sin_phi_sq = sin(lat) ** 2
        return A[0] * (1 + A[1] * sin_phi_sq + A[2] * sin_phi_sq**2) + (A[3] + A[4] * sin_phi_sq) * h + A[5] * h**2

    @classmethod
    def radii_of_curvature(cls, lat: float) -> tuple[float]:
        '''
        Compute the radii of curvature of the Earth at the given latitude.
        WGS84 parameters are used by default.
        '''
        one_minus_e_sq_sin_phi_sq = 1 - cls.e_squared * sin(lat) ** 2
        N = cls.semimajor_axis / sqrt(one_minus_e_sq_sin_phi_sq)
        M = N * (1 - cls.e_squared) / one_minus_e_sq_sin_phi_sq
        return N, M

    def compensate_errors_and_compute_params(
        self, acc: np.ndarray, omega: np.ndarray, return_params: bool = True
    ) -> tuple[(np.ndarray,) * 2] | tuple[(np.ndarray,) * 2, (float,) * 3]:
        '''Compensate for deterministic errors and compute the Earth parameters'''

        # Compute gravity
        g = self.gravity(self.lat, self.h)

        # If biases depend on gravity, convert them to floats
        accel_bias = self.accel_bias(g) if callable(self.accel_bias) else self.accel_bias

        # Compensate for deterministic errors
        acc = self.accel_inv_error_matrix @ (acc - accel_bias)
        omega = self.gyro_inv_error_matrix @ (omega - self.gyro_bias)

        if not return_params:
            return acc, omega

        # Calculate the radii of curvature - the prime vertical N and meridian M
        N, M = self.radii_of_curvature(self.lat)
        return acc, omega, g, N, M

    def align(self, acc: np.ndarray, omega: np.ndarray, timestamp: float) -> None:
        '''Compute the alignment of the IMU'''

        if self.alignment_complete:
            raise RuntimeError('align method called after alignment is complete')

        if timestamp < self.alignment_time:
            self.alignment_it += 1

            # Compute the error corrected accel and gyro measurements
            acc, omega = self.compensate_errors_and_compute_params(acc, omega, return_params=False)

            # Add to the running means
            self.alignment_acc_mean += acc
            self.alignment_omega_mean += omega

        else:
            # Alignment completed, compute final means and rotation parameters
            self.alignment_acc_mean /= self.alignment_it
            self.alignment_omega_mean /= self.alignment_it

            # Compute gravity
            g = self.gravity(self.lat, self.h)

            # Compute roll, pitch, and azimuth based on means of measurements during static alignment
            self.roll = -np.sign(self.alignment_acc_mean[2, 0]) * asin(self.alignment_acc_mean[0, 0] / g)
            self.pitch = np.sign(self.alignment_acc_mean[2, 0]) * asin(self.alignment_acc_mean[1, 0] / g)

            # Rotate the gyroscope measurements to the level plane before performing gyro compassing
            levelled_alignment_omega_mean = self.Rx(self.pitch) @ self.Ry(self.roll) @ self.alignment_omega_mean
            self.azimuth = atan(-levelled_alignment_omega_mean[0, 0] / levelled_alignment_omega_mean[1, 0])

            # Compute rotation matrix and the associated quaternion
            R_b2l = self.get_rotation_matrix(self.roll, self.pitch, self.azimuth)
            self.quat = self.matrix_to_normalized_quaternion(R_b2l)
            self.R_b2l = self.quaternion_to_matrix(self.quat)

            # Set the initial attitude
            self.initial_attitude = np.array([self.roll, self.pitch, self.azimuth])

            # Flag alignment as complete
            self.alignment_complete = True

    def angular_velocity_compensation(self, N: float, M: float, omega: np.ndarray, delta_t: float) -> np.ndarray:
        '''Compensate for the LLF transportation rate and the Earth's rotation'''

        # Compute the rotation between the inertial frame and LLF as seen in the body frame
        N_plus_h = N + self.h
        omega_lib = self.R_b2l.T @ np.array(
            [
                [-self.v_llf[1, 0] / (M + self.h)],
                [self.v_llf[0, 0] / N_plus_h + self.omega_e * cos(self.lat)],
                [self.v_llf[0, 0] * tan(self.lat) / N_plus_h + self.omega_e * sin(self.lat)],
            ]
        )

        # Compute and return the angular changes of the body
        theta_lib = omega_lib * delta_t
        theta_bib = omega * delta_t
        theta_blb = theta_bib - theta_lib
        return theta_blb

    def attitude_integration(self, d_theta: np.ndarray) -> None:
        '''Update the quaternion and rotation matrices and compute roll, pitch, and azimuth'''

        d_theta_x, d_theta_y, d_theta_z = d_theta

        # Update the quaternion based on the angular increments
        quat_update_matrix = np.array(
            [
                [0, d_theta_z, -d_theta_y, d_theta_x],
                [-d_theta_z, 0, d_theta_x, d_theta_y],
                [d_theta_y, -d_theta_x, 0, d_theta_z],
                [-d_theta_x, -d_theta_y, -d_theta_z, 0],
            ]
        )
        self.quat += 0.5 * quat_update_matrix @ self.quat

        # Normalize the quaternion
        self.quat /= sqrt(self.quat[0, 0] ** 2 + self.quat[1, 0] ** 2 + self.quat[2, 0] ** 2 + self.quat[3, 0] ** 2)

        # Compute the associated rotation matrix
        self.R_b2l = self.quaternion_to_matrix(self.quat)

        # Compute roll, pitch, and azimuth from the rotation matrix
        self.roll = atan(-self.R_b2l[2, 0] / self.R_b2l[2, 2])
        self.pitch = atan(self.R_b2l[2, 1] / sqrt(self.R_b2l[0, 1] ** 2 + self.R_b2l[1, 1] ** 2))
        self.azimuth = atan(self.R_b2l[0, 1] / self.R_b2l[1, 1])

    def v_and_r_integration(self, acc: np.ndarray, delta_t: float, g: float, N: float, M: float) -> None:
        '''Compute the velocity increments and update the LLF velocity and the position'''

        # Compute omega_lel in skew-symmetric form
        N_plus_h = N + self.h
        omega_lel = np.array(
            [
                -self.v_llf[1, 0] / (M + self.h),
                self.v_llf[0, 0] / N_plus_h,
                self.v_llf[0, 0] * tan(self.lat) / N_plus_h,
            ]
        )
        Omega_lel = self.vec_to_skew(omega_lel)

        # Compute omega_eil in skew-symmetric form
        omega_eil = np.array([0, self.omega_e * cos(self.lat), self.omega_e * sin(self.lat)])
        Omega_eil = self.vec_to_skew(omega_eil)

        # Gravity vector
        g_vec = np.array([[0], [0], [-g]])

        # Compute the change in velocity in the LLF
        delta_v_llf = (self.R_b2l @ acc - (2 * Omega_eil + Omega_lel) @ self.v_llf + g_vec) * delta_t

        # Update the LLF velocity
        new_v_llf = self.v_llf + 0.5 * (self.prev_delta_v_llf + delta_v_llf)
        self.prev_delta_v_llf = delta_v_llf

        # Update the altitude, latitude, and longitude
        mean_v_llf = 0.5 * (self.v_llf + new_v_llf)
        self.long += delta_t * mean_v_llf[0, 0] / ((N + self.h) * cos(self.lat))
        self.lat += delta_t * mean_v_llf[1, 0] / (M + self.h)
        self.h += delta_t * mean_v_llf[2, 0]

        # Save the new LLF velocity
        self.v_llf = new_v_llf

    def process_measurement(self, measurement: np.ndarray) -> None:
        '''Process a measurement from the IMU'''

        # In case we get a non-zero start time, shift the time steps back to the time since the first measurment
        if self.start_time is None:
            self.start_time = measurement[0]
        self.timestamp = measurement[0] - self.start_time
        omega = measurement[1:4].reshape((3, 1))
        acc = measurement[4:].reshape((3, 1))

        delta_t = self.timestamp - self.prev_time
        self.prev_time = self.timestamp

        # If alignment is not complete, call align
        if not self.alignment_complete:
            self.align(acc, omega, self.timestamp)
            return

        # Compensate for deterministic errors and compute the Earth parameters
        acc, omega, g, N, M = self.compensate_errors_and_compute_params(acc, omega)

        # Compensate for Earth's rotation and the motion of the LLF
        theta_blb = self.angular_velocity_compensation(N, M, omega, delta_t)

        # Integrate to determine attitude (roll, pitch, azimuth)
        self.attitude_integration(theta_blb.reshape(3))

        # Integrate to update position and LLF velocity
        self.v_and_r_integration(acc, delta_t, g, N, M)

        # Update the position errors
        # This assumes that the current LLF is not significantly different from the initial LLF
        # This is a reasonable assumption for this project as the IMU is stationary
        self.position_errors += self.v_llf * delta_t

    def get_params(
        self, get_labels: bool = False, degrees: bool = True
    ) -> tuple[(str,) * 14] | tuple[(float,) * 13, bool]:
        '''
        Return the current navigation parameters

        Returns column labels if get_labels is True
        '''
        if get_labels:
            return (
                'time',
                'latitude',
                'longitude',
                'height',
                'velocity east',
                'velocity north',
                'velocity up',
                'roll',
                'pitch',
                'azimuth',
                'delta v east',
                'delta v north',
                'delta v up',
                'is aligning',
            )
        if degrees:
            return (
                self.timestamp,
                self.lat * RAD_TO_DEG,
                self.long * RAD_TO_DEG,
                self.h,
                *self.v_llf.reshape(3),
                self.roll * RAD_TO_DEG,
                self.pitch * RAD_TO_DEG,
                self.azimuth * RAD_TO_DEG,
                *self.position_errors.reshape(3),
                not self.alignment_complete,
            )
        return (
            self.timestamp,
            self.lat,
            self.long,
            self.h,
            *self.v_llf.reshape(3),
            self.roll,
            self.pitch,
            self.azimuth,
            *self.position_errors.reshape(3),
            not self.alignment_complete,
        )


def plot_results(
    timestamps: np.ndarray,
    data: np.ndarray,
    y_labels: list,
    y_lims: list,
    y_ticks: list,
    title: str,
    save: bool = True,
) -> None:
    '''Helper function to plot results'''

    # If necessary, import the required modules
    if 'plt' not in globals():
        global plt, ScalarFormatter, AutoMinorLocator
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter, AutoMinorLocator

    plt.figure(figsize=(10, 11))
    for i in range(3):
        # Plot the results
        plt.subplot(3, 1, i + 1)
        plt.plot(timestamps, data[:, i], linewidth=1)

        # Set the axis ticks
        plt.yticks(np.linspace(*y_lims[i], y_ticks if isinstance(y_ticks, int) else y_ticks[i]))
        plt.xticks(range(0, 1000, 100))

        # Format the grid lines
        plt.minorticks_on()
        ax = plt.gca()
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(True, which='major', alpha=0.6, linestyle='--')
        plt.grid(True, which='minor', alpha=0.4, linestyle=':')

        # Add axes labels
        if i == 2:
            plt.xlabel('Time (s)')
        plt.ylabel(y_labels[i])

        # Format the y-axis tick labels to remove the constant
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    if save:
        plt.savefig(title, dpi=300)
    else:
        plt.show()


def main(save_plots: bool = False, save_results_csv: bool = False) -> None:
    '''Run the mechanization and plot the results'''

    import csv
    import time

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

    alignment_time = 120  # s

    # Instantiate the model
    INS = INSMechanization(
        h,
        lat,
        long,
        acc_bias,
        gyro_bias,
        scale_factor,
        scale_factor,
        nonorthogonality,
        nonorthogonality,
        alignment_time,
    )

    # Run the module one measurement at a time
    results = []
    t0 = time.perf_counter()
    for measurement in data:
        INS.process_measurement(measurement)
        results.append(INS.get_params())
    print(f'Mechanization completed in {time.perf_counter() - t0:.3f} seconds')

    # Save the results in csv format
    if save_results_csv:
        with open('results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(INS.get_params(get_labels=True))
            writer.writerows(results)

    # Exclude results during alignment
    results = np.array([r[:-1] for r in results if not r[-1]])

    # Shift the timestamps to start at 0
    timestamps = results[:, 0] - results[0, 0]

    # Plot the position
    plot_results(
        timestamps,
        results[:, 1:4],
        (r'$\phi$ (deg)', r'$\lambda$ (deg)', 'h (m)'),
        ((51.07, 51.08), (-114.134, -114.128), (1100, 1350)),
        6,
        'img/position.png',
        save_plots,
    )

    # Plot the position errors
    plot_results(
        timestamps,
        results[:, 10:],
        (r'$\delta$ E (m)', r'$\delta$ N (m)', r'$\delta$ Up (m)'),
        ((0, 400), (-1000, 0), (0, 200)),
        6,
        'img/position_errors.png',
        save_plots,
    )

    # Plot the velocity errors
    plot_results(
        timestamps,
        results[:, 4:7],
        (r'$\delta V_E$ (m/s)', r'$\delta V_N$ (m/s)', r'$\delta V_{UP}$ (m/s)'),
        ((0, 1.2), (-3.5, 0), (0, 0.6)),
        (7, 8, 7),
        'img/velocity_errors.png',
        save_plots,
    )

    # Plot the attitude errors
    plot_results(
        timestamps,
        results[:, 7:10] - INS.initial_attitude * RAD_TO_DEG,
        (r'$\delta r$ (deg)', r'$\delta p$ (deg)', r'$\delta A$ (deg)'),
        ((-0.008, 0.004), (0, 0.06), (-0.02, 0)),
        (7, 7, 6),
        'img/attitude_errors.png',
        save_plots,
    )


if __name__ == '__main__':
    main(save_plots=True, save_results_csv=True)
