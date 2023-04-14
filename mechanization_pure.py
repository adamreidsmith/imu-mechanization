from math import copysign, sqrt, sin, cos, tan, asin, atan, pi
from typing import Callable


RAD_TO_DEG = 180 / pi


# need to compute errors with ARW, etc.


class INSMechanization:
    '''
    Class to compute mechanization of IMU data.
    All values must be in SI base units or SI derived units.
    '''

    e_squared = 6.69438e-3  # squared WGS84 eccentricity of the Earth
    semimajor_axis = 6378137  # WGS84 semi-major axis of the Earth (m)
    omega_e = 7.292115147e-5  # angular velocity of the Earth (rad / s)
    i3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    def __init__(
        self,
        h0: float,  # initial height
        lat0: float,  # initial latitude
        long0: float,  # initial longitude
        accel_bias: float | Callable = 0,  # accel bias
        gyro_bias: float = 0,  # gyro bias
        accel_sf: float | list = 0,  # accel scale factor matrix
        gyro_sf: float | list = 0,  # gyro scale factor matrix
        accel_no: list = ((0, 0, 0), (0, 0, 0), (0, 0, 0)),  # accel non-orthogonality
        gyro_no: list = ((0, 0, 0), (0, 0, 0), (0, 0, 0)),  # gyro non-orthogonality
        alignment_time: float = 0,  # alignment time in seconds
    ) -> None:
        self.h = h0
        self.lat = lat0
        self.long = long0
        self.accel_bias = accel_bias
        self.gyro_bias = gyro_bias
        accel_sf = (
            ((accel_sf, 0, 0), (0, accel_sf, 0), (0, accel_sf, 0)) if isinstance(accel_sf, (float, int)) else accel_sf
        )
        gyro_sf = ((gyro_sf, 0, 0), (0, gyro_sf, 0), (0, gyro_sf, 0)) if isinstance(gyro_sf, (float, int)) else gyro_sf
        self.accel_inv_error_matrix = self.matinv(self.matsum(self.matsum(self.i3, accel_sf), accel_no))
        self.gyro_inv_error_matrix = self.matinv(self.matsum(self.matsum(self.i3, gyro_sf), gyro_no))
        self.quat = None  # rotation quaternion; inititalized when self.align is called
        self.R_b2l = None  # rotation matrix; inititalized when self.align is called
        self.v_llf = [0, 0, 0]  # initial ENU velocities
        self.prev_delta_v_llf = [0, 0, 0]  # change in v in LLF at previous t
        self.prev_time = 0  # previous time, used to determine delta_t
        self.roll = self.pitch = self.azimuth = 0  # initialize to store orientation
        self.timestamp = 0  # store the current timestamp
        self.start_time = None  # start time of the mechanization

        # Alignment specific attributes
        self.alignment_time = alignment_time
        self.alignment_complete = False  # flag to determine if alignment is completed
        self.alignment_acc_mean = [0, 0, 0]  # running mean for accel alignment
        self.alignment_omega_mean = [0, 0, 0]  # running mean for gyro alignment
        self.alignment_it = 0  # couter to keep track of alignment iterations

        # Values for error tracking
        self.initial_attitude = None  # initial roll, pitch, azimuth as computed by alignment
        self.position_errors = [0, 0, 0]  # errors in position in the initial LLF

    @staticmethod
    def matinv(m):
        '''Inverse of a 3x3 matrix'''

        a, b, c = m[0]
        d, e, f = m[1]
        g, h, i = m[2]

        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

        dinv = 1 / det
        return [
            [
                (e * i - f * h) * dinv,
                (c * h - b * i) * dinv,
                (b * f - c * e) * dinv,
            ],
            [
                (f * g - d * i) * dinv,
                (a * i - c * g) * dinv,
                (c * d - a * f) * dinv,
            ],
            [
                (d * h - e * g) * dinv,
                (b * g - a * h) * dinv,
                (a * e - b * d) * dinv,
            ],
        ]

    @staticmethod
    def matsum(m1, m2):
        '''Sum of two 3x3 matrices'''

        return [
            [m1[0][0] + m2[0][0], m1[0][1] + m2[0][1], m1[0][2] + m2[0][2]],
            [m1[1][0] + m2[1][0], m1[1][1] + m2[1][1], m1[1][2] + m2[1][2]],
            [m1[2][0] + m2[2][0], m1[2][1] + m2[2][1], m1[2][2] + m2[2][2]],
        ]

    @staticmethod
    def matvec(m, v):
        '''3x3 matrix times a 3x1 vector (passed in with shape (3,))'''

        return [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]

    @staticmethod
    def matvec4(m, v):
        '''4x4 matrix times a 4x1 vector (passed in with shape (4,))'''

        return [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
            m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3],
        ]

    @staticmethod
    def T3(m):
        '''Transpose of a 3x3 matrix'''

        a, b, c = m[0]
        d, e, f = m[1]
        g, h, i = m[2]
        return [[a, d, g], [b, e, h], [c, f, i]]

    @staticmethod
    def matprod(A, B):
        '''Product of 2 3x3 matrices'''

        return [
            [
                A[0][0] * B[0][0] + A[0][1] * B[1][0] + A[0][2] * B[2][0],
                A[0][0] * B[0][1] + A[0][1] * B[1][1] + A[0][2] * B[2][1],
                A[0][0] * B[0][2] + A[0][1] * B[1][2] + A[0][2] * B[2][2],
            ],
            [
                A[1][0] * B[0][0] + A[1][1] * B[1][0] + A[1][2] * B[2][0],
                A[1][0] * B[0][1] + A[1][1] * B[1][1] + A[1][2] * B[2][1],
                A[1][0] * B[0][2] + A[1][1] * B[1][2] + A[1][2] * B[2][2],
            ],
            [
                A[2][0] * B[0][0] + A[2][1] * B[1][0] + A[2][2] * B[2][0],
                A[2][0] * B[0][1] + A[2][1] * B[1][1] + A[2][2] * B[2][1],
                A[2][0] * B[0][2] + A[2][1] * B[1][2] + A[2][2] * B[2][2],
            ],
        ]

    @classmethod
    def get_rotation_matrix(cls, r, p, A):
        '''Compute the rotation matrix to rotate the body frame to the LLF from the Euler angles'''

        return cls.matprod(cls.Rz(-A), cls.matprod(cls.Rx(p), cls.Ry(r)))

    @staticmethod
    def matrix_to_quaternion(R):
        '''Convert a rotation matrix to a normalized quaternion'''

        q4 = sqrt(1 + R[0][0] + R[1][1] + R[2][2]) / 2
        four_q4 = 4 * q4
        q1 = (R[2][1] - R[1][2]) / four_q4
        q2 = (R[0][2] - R[2][0]) / four_q4
        q3 = (R[1][0] - R[0][1]) / four_q4
        q_norm = sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
        return [q1 / q_norm, q2 / q_norm, q3 / q_norm, q4 / q_norm]

    @staticmethod
    def quaternion_to_matrix(q):
        '''Convert a normalized quaternion to a rotation matrix'''

        q1, q2, q3, q4 = q
        qq1 = q1 * q1
        qq2 = q2 * q2
        qq3 = q3 * q3
        qq4 = q4 * q4
        q1q2 = q1 * q2
        q3q4 = q3 * q4
        q1q3 = q1 * q3
        q2q4 = q2 * q4
        q1q4 = q1 * q4
        q2q3 = q2 * q3

        return [
            [qq1 - qq2 - qq3 + qq4, 2 * (q1q2 - q3q4), 2 * (q1q3 + q2q4)],
            [2 * (q1q2 + q3q4), qq2 + qq4 - qq1 - qq3, 2 * (q2q3 - q1q4)],
            [2 * (q1q3 - q2q4), 2 * (q2q3 + q1q4), qq3 + qq4 - qq1 - qq2],
        ]

    @staticmethod
    def vec_to_skew(v):
        '''Convert a vector to its skew-symmetric representation'''

        v1, v2, v3 = v
        return [[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]]

    @staticmethod
    def Rx(theta):
        '''Returns the rotation matrix about the x-axis by angle theta using the sign convention in Wikipedia'''
        costheta = cos(theta)
        sintheta = sin(theta)
        return [[1, 0, 0], [0, costheta, -sintheta], [0, sintheta, costheta]]

    @staticmethod
    def Ry(theta):
        '''Returns the rotation matrix about the x-axis by angle theta using the sign convention in Wikipedia'''
        costheta = cos(theta)
        sintheta = sin(theta)
        return [[costheta, 0, sintheta], [0, 1, 0], [-sintheta, 0, costheta]]

    @staticmethod
    def Rz(theta):
        '''Returns the rotation matrix about the z-axis by angle theta using the sign convention in Wikipedia'''
        costheta = cos(theta)
        sintheta = sin(theta)
        return [[costheta, -sintheta, 0], [sintheta, costheta, 0], [0, 0, 1]]

    @staticmethod
    def gravity(lat, h):
        '''Compute the gravity vector given the latitude and height'''

        sin_phi_sq = sin(lat) ** 2
        return (
            9.7803267715 * (1 + 0.0052790414 * sin_phi_sq + 2.32718e-5 * sin_phi_sq * sin_phi_sq)
            - (3.087691089e-6 - 4.397731e-9 * sin_phi_sq) * h
            + 7.21e-13 * h * h
        )

    @classmethod
    def radii_of_curvature(cls, lat):
        '''
        Compute the radii of curvature of the Earth at the given latitude.
        WGS84 parameters are used by default.
        '''
        one_minus_e_sq_sin_phi_sq = 1 - cls.e_squared * sin(lat) ** 2
        N = cls.semimajor_axis / sqrt(one_minus_e_sq_sin_phi_sq)
        M = N * (1 - cls.e_squared) / one_minus_e_sq_sin_phi_sq
        return N, M

    def compensate_errors_and_compute_params(self, acc, omega, return_params=True):
        '''Compensate for deterministic errors and compute the Earth parameters'''

        # Compute gravity
        g = self.gravity(self.lat, self.h)

        # If biases depend on gravity, convert them to floats
        accel_bias = self.accel_bias(g) if callable(self.accel_bias) else self.accel_bias

        # Compensate for deterministic errors
        acc = self.matvec(self.accel_inv_error_matrix, [acc[0] - accel_bias, acc[1] - accel_bias, acc[2] - accel_bias])
        omega = self.matvec(
            self.gyro_inv_error_matrix,
            [omega[0] - self.gyro_bias, omega[1] - self.gyro_bias, omega[2] - self.gyro_bias],
        )

        if not return_params:
            return acc, omega

        # Calculate the radii of curvature - the prime vertical N and meridian M
        N, M = self.radii_of_curvature(self.lat)
        return acc, omega, g, N, M

    def align(self, acc, omega, timestamp):
        '''Compute the alignment of the IMU'''

        if self.alignment_complete:
            raise RuntimeError('align method called after alignment is complete')

        if timestamp < self.alignment_time:
            self.alignment_it += 1

            # Compute the error corrected accel and gyro measurements
            acc, omega = self.compensate_errors_and_compute_params(acc, omega, return_params=False)

            # Add to the running means
            self.alignment_acc_mean[0] += acc[0]
            self.alignment_acc_mean[1] += acc[1]
            self.alignment_acc_mean[2] += acc[2]
            self.alignment_omega_mean[0] += omega[0]
            self.alignment_omega_mean[1] += omega[1]
            self.alignment_omega_mean[2] += omega[2]

        else:
            # Alignment completed, compute final means and rotation parameters
            self.alignment_acc_mean[0] /= self.alignment_it
            self.alignment_acc_mean[1] /= self.alignment_it
            self.alignment_acc_mean[2] /= self.alignment_it
            self.alignment_omega_mean[0] /= self.alignment_it
            self.alignment_omega_mean[1] /= self.alignment_it
            self.alignment_omega_mean[2] /= self.alignment_it

            # Compute gravity
            g = self.gravity(self.lat, self.h)

            # Compute roll, pitch, and azimuth based on means of measurements during static alignment
            self.roll = -copysign(1, self.alignment_acc_mean[2]) * asin(self.alignment_acc_mean[0] / g)
            self.pitch = copysign(1, self.alignment_acc_mean[2]) * asin(self.alignment_acc_mean[1] / g)

            # Rotate the gyroscope measurements to the level plane before performing gyro compassing
            levelled_alignment_omega_mean = self.matvec(
                self.Rx(self.pitch), self.matvec(self.Ry(self.roll), self.alignment_omega_mean)
            )
            self.azimuth = atan(-levelled_alignment_omega_mean[0] / levelled_alignment_omega_mean[1])

            # Compute rotation matrix and the associated quaternion
            R_b2l = self.get_rotation_matrix(self.roll, self.pitch, self.azimuth)
            self.quat = self.matrix_to_quaternion(R_b2l)
            self.R_b2l = self.quaternion_to_matrix(self.quat)

            # Set the initial attitude
            self.initial_attitude = (self.roll, self.pitch, self.azimuth)

            # Flag alignment as complete
            self.alignment_complete = True

    def angular_velocity_compensation(self, N, M, omega, delta_t):
        '''Compensate for the LLF transportation rate and the Earth's rotation'''

        # Compute the rotation between the inertial frame and LLF as seen in the body frame
        N_plus_h = N + self.h
        R_b2lT = self.T3(self.R_b2l)
        omega_lib = self.matvec(
            R_b2lT,
            [
                -self.v_llf[1] / (M + self.h),
                self.v_llf[0] / N_plus_h + self.omega_e * cos(self.lat),
                self.v_llf[0] * tan(self.lat) / N_plus_h + self.omega_e * sin(self.lat),
            ],
        )

        # Compute and return the angular changes of the body
        theta_lib = [omega_lib[0] * delta_t, omega_lib[1] * delta_t, omega_lib[2] * delta_t]
        theta_bib = [omega[0] * delta_t, omega[1] * delta_t, omega[2] * delta_t]
        theta_blb = [theta_bib[0] - theta_lib[0], theta_bib[1] - theta_lib[1], theta_bib[2] - theta_lib[2]]
        return theta_blb

    def attitude_integration(self, d_theta_x, d_theta_y, d_theta_z):
        '''Update the quaternion and rotation matrices and compute roll, pitch, and azimuth'''

        # Update the quaternion based on the angular increments
        quat_update_matrix = [
            [0, d_theta_z, -d_theta_y, d_theta_x],
            [-d_theta_z, 0, d_theta_x, d_theta_y],
            [d_theta_y, -d_theta_x, 0, d_theta_z],
            [-d_theta_x, -d_theta_y, -d_theta_z, 0],
        ]
        quat_update = self.matvec4(quat_update_matrix, self.quat)
        self.quat[0] += 0.5 * quat_update[0]
        self.quat[1] += 0.5 * quat_update[1]
        self.quat[2] += 0.5 * quat_update[2]
        self.quat[3] += 0.5 * quat_update[3]

        # Normalize the quaternion
        q_norm = sqrt(self.quat[0] ** 2 + self.quat[1] ** 2 + self.quat[2] ** 2 + self.quat[3] ** 2)
        self.quat = [self.quat[0] / q_norm, self.quat[1] / q_norm, self.quat[2] / q_norm, self.quat[3] / q_norm]

        # Compute the associated rotation matrix
        self.R_b2l = self.quaternion_to_matrix(self.quat)

        # Compute roll, pitch, and azimuth from the rotation matrix
        self.roll = atan(-self.R_b2l[2][0] / self.R_b2l[2][2])
        self.pitch = atan(self.R_b2l[2][1] / sqrt(self.R_b2l[0][1] ** 2 + self.R_b2l[1][1] ** 2))
        self.azimuth = atan(self.R_b2l[0][1] / self.R_b2l[1][1])

    def v_and_r_integration(self, acc, delta_t, g, N, M):
        '''Compute the velocity increments and update the LLF velocity and the position'''

        # Rotate acceleration to the LLF
        acc_llf = self.matvec(self.R_b2l, acc)

        # Compute omega_lel in skew-symmetric form
        N_plus_h = N + self.h
        omega_lel = [
            -self.v_llf[1] / (M + self.h),
            self.v_llf[0] / N_plus_h,
            self.v_llf[0] * tan(self.lat) / N_plus_h,
        ]
        Omega_lel = self.vec_to_skew(omega_lel)

        # Compute omega_eil in skew-symmetric form
        omega_eil = [0, self.omega_e * cos(self.lat), self.omega_e * sin(self.lat)]
        Omega_eil = self.vec_to_skew(omega_eil)

        # Compute the change in velocity in the LLF
        temp = [
            [
                2 * Omega_eil[0][0] + Omega_lel[0][0],
                2 * Omega_eil[0][1] + Omega_lel[0][1],
                2 * Omega_eil[0][2] + Omega_lel[0][2],
            ],
            [
                2 * Omega_eil[1][0] + Omega_lel[1][0],
                2 * Omega_eil[1][1] + Omega_lel[1][1],
                2 * Omega_eil[1][2] + Omega_lel[1][2],
            ],
            [
                2 * Omega_eil[2][0] + Omega_lel[2][0],
                2 * Omega_eil[2][1] + Omega_lel[2][1],
                2 * Omega_eil[2][2] + Omega_lel[2][2],
            ],
        ]
        temp = self.matvec(temp, self.v_llf)

        delta_v_llf = [0.0, 0.0, 0.0]
        delta_v_llf[0] = (acc_llf[0] - temp[0]) * delta_t
        delta_v_llf[1] = (acc_llf[1] - temp[1]) * delta_t
        delta_v_llf[2] = (acc_llf[2] - temp[2] - g) * delta_t

        # Update the LLF velocity
        new_v_llf = [0.0, 0.0, 0.0]
        new_v_llf[0] = self.v_llf[0] + 0.5 * (self.prev_delta_v_llf[0] + delta_v_llf[0])
        new_v_llf[1] = self.v_llf[1] + 0.5 * (self.prev_delta_v_llf[1] + delta_v_llf[1])
        new_v_llf[2] = self.v_llf[2] + 0.5 * (self.prev_delta_v_llf[2] + delta_v_llf[2])
        self.prev_delta_v_llf = delta_v_llf

        # Update the altitude, latitude, and longitude
        mean_v_llf = [0, 0, 0]
        mean_v_llf[0] = 0.5 * (self.v_llf[0] + new_v_llf[0])
        mean_v_llf[1] = 0.5 * (self.v_llf[1] + new_v_llf[1])
        mean_v_llf[2] = 0.5 * (self.v_llf[2] + new_v_llf[2])

        self.long += delta_t * mean_v_llf[0] / ((N + self.h) * cos(self.lat))
        self.lat += delta_t * mean_v_llf[1] / (M + self.h)
        self.h += delta_t * mean_v_llf[2]

        # Save the new LLF velocity
        self.v_llf = new_v_llf

    def process_measurement(self, measurement):
        '''Process a measurement from the IMU'''

        # In case we get a non-zero start time, shift the time steps back to the time since the first measurment
        if self.start_time is None:
            self.start_time = measurement[0]
        self.timestamp = measurement[0] - self.start_time
        omega = measurement[1:4]
        acc = measurement[4:]

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
        d_theta_x, d_theta_y, d_theta_z = theta_blb
        self.attitude_integration(d_theta_x, d_theta_y, d_theta_z)

        # Integrate to update position and LLF velocity
        self.v_and_r_integration(acc, delta_t, g, N, M)

        # Update the position errors
        # This assumes that the current LLF is not significantly different from the initial LLF
        # This is a reasonable assumption for this project as the IMU is stationary
        self.position_errors[0] += self.v_llf[0] * delta_t
        self.position_errors[1] += self.v_llf[1] * delta_t
        self.position_errors[2] += self.v_llf[2] * delta_t

    def get_params(self, get_labels=False, degrees=True):
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
                'isAligning',
            )
        if degrees:
            return (
                self.timestamp,
                self.lat * RAD_TO_DEG,
                self.long * RAD_TO_DEG,
                self.h,
                *self.v_llf,
                self.roll * RAD_TO_DEG,
                self.pitch * RAD_TO_DEG,
                self.azimuth * RAD_TO_DEG,
                *self.position_errors,
                not self.alignment_complete,
            )
        return (
            self.timestamp,
            self.lat,
            self.long,
            self.h,
            *self.v_llf,
            self.roll,
            self.pitch,
            self.azimuth,
            *self.position_errors,
            not self.alignment_complete,
        )


def plot_results(timestamps, data, y_labels, y_lims, y_ticks, title, save=True):
    '''Helper function to plot results'''

    # If necessary, import the required modules
    if 'plt' not in globals():
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
    if 'np' not in globals():
        import numpy as np

    plt.figure(figsize=(11, 9))
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
    if save:
        plt.savefig(title, dpi=300)
    else:
        plt.show()


def main():
    '''Run the mechanization and plot the results'''
    import csv
    import time
    from math import pi
    import numpy as np

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

    # # Save the results in csv format
    # with open('results.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(INS.get_params(get_labels=True))
    #     writer.writerows(results)

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
        False,
    )

    # Plot the position errors
    plot_results(
        timestamps,
        results[:, 10:],
        (r'$\delta$ E (m)', r'$\delta$ N (m)', r'$\delta$ Up (m)'),
        ((0, 400), (-1000, 0), (0, 200)),
        6,
        'img/position_errors.png',
        False,
    )

    # Plot the velocity errors
    plot_results(
        timestamps,
        results[:, 4:7],
        (r'$\delta V_E$ (m/s)', r'$\delta V_N$ (m/s)', r'$\delta V_{UP}$ (m/s)'),
        ((0, 1.2), (-3.5, 0), (0, 0.6)),
        (7, 8, 7),
        'img/velocity_errors.png',
        False,
    )
    # Plot the attitude errors
    plot_results(
        timestamps,
        results[:, 7:10] - np.array(INS.initial_attitude) * RAD_TO_DEG,
        (r'$\delta r$ (deg)', r'$\delta p$ (deg)', r'$\delta A$ (deg)'),
        ((-0.008, 0.004), (0, 0.06), (-0.02, 0)),
        (7, 7, 6),
        'img/attitude_erros.png',
        False,
    )


if __name__ == '__main__':
    main()
