from math import sqrt, sin, cos, tan, asin, atan, pi
import numpy as np
from typing import Callable

RAD_TO_DEG = 180 / pi


# need to compute errors with ARW, etc.


class INSMechanization:

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
        vrw: float = 0.0,  # velocity random walk
        arw: float = 0.0,  # angle random walk
        accel_corr_time: float = 0.0,  # accel correlation time
        gyro_corr_time: float = 0.0,  # gyro correlation time
        accel_bias_instability: float | Callable = 0.0,  # accel bias instability
        gyro_bias_instability: float = 0.0,  # gyro bias instability
        alignment_time: float = 0,  # alignment time in seconds
    ) -> None:
        self.h = h0
        self.lat = lat0
        self.long = long0
        self.accel_bias = accel_bias
        self.gyro_bias = gyro_bias
        self.arw = arw
        self.vrw = vrw
        self.accel_corr_time = accel_corr_time
        self.gyro_corr_time = gyro_corr_time
        self.accel_bias_instability = accel_bias_instability
        self.gyro_bias_instability = gyro_bias_instability
        accel_sf = accel_sf * np.eye(3) if isinstance(accel_sf, (float, int)) else accel_sf
        gyro_sf = gyro_sf * np.eye(3) if isinstance(gyro_sf, (float, int)) else gyro_sf
        self.accel_inv_error_matrix = np.linalg.inv(np.eye(3) + accel_sf + accel_no)
        self.gyro_inv_error_matrix = np.linalg.inv(np.eye(3) + gyro_sf + gyro_no)
        self.quat = None  # rotation quaternion; inititalized when alignment completes
        self.R_b2l = None  # rotation matrix; inititalized when alignment completes
        self.v_llf = np.zeros((3, 1))  # initial ENU velocities
        self.prev_delta_v_llf = np.zeros((3, 1))  # change in v in LLF at previous t
        self.prev_time = None  # previous time, used to determine delta_t
        self.roll = self.pitch = self.azimuth = 0  # initialize to store orientation
        self.timestamp = 0  # store the current timestamp
        self.start_time = None  # start time of the mechanization

        # Alignment specific attributes
        self.alignment_time = alignment_time
        self.alignment_complete = False  # flag to determine if alignment is completed
        self.alignment_acc_mean = np.zeros((3, 1))  # running mean for accel alignment
        self.alignment_omega_mean = np.zeros((3, 1))  # running mean for gyro alignment
        self.alignment_it = 0  # couter to keep track of alignment iterations
        self.post_alignment_roll_error = -1  # roll error after alignment
        self.post_alignment_pitch_error = -1  # pitch error after alignment
        self.post_alignment_azimuth_error = -1  # azimuth error after alignment

        # Errors
        self.roll_error = 0
        self.pitch_error = 0
        self.azimuth_error = 0

    @staticmethod
    def get_rotation_matrix(r, p, A):
        '''Compute the rotation matrix to rotate the body frame to the LLF from the Euler angles'''

        cosr = cos(r)
        cosp = cos(p)
        cosa = cos(A)
        sinr = sin(r)
        sinp = sin(p)
        sina = sin(A)
        cosacosr = cosa * cosr
        sinasinr = sina * sinr
        sinacosr = sina * cosr
        cosasinr = cosa * sinr
        return np.array(
            [
                [
                    cosacosr + sinasinr * sinp,
                    sina * cosp,
                    cosasinr - sinacosr * sinp,
                ],
                [
                    cosasinr * sinp - sinacosr,
                    cosa * cosp,
                    -sinasinr - cosacosr * sinp,
                ],
                [-cosp * sinr, sinp, cosp * cosr],
            ]
        )

    @staticmethod
    def matrix_to_normalized_quaternion(R):
        '''Convert a rotation matrix to a normalized quaternion'''

        q4 = sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        four_q4 = 4 * q4
        q1 = (R[2, 1] - R[1, 2]) / four_q4
        q2 = (R[0, 2] - R[2, 0]) / four_q4
        q3 = (R[1, 0] - R[0, 1]) / four_q4
        q_norm = sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
        return np.array([[q1 / q_norm], [q2 / q_norm], [q3 / q_norm], [q4 / q_norm]])

    @staticmethod
    def quaternion_to_matrix(q):
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
    def vec_to_skew(v):
        '''Convert a vector to its skew-symmetric representation'''

        v1, v2, v3 = v.reshape(3)
        return np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])

    @staticmethod
    def gravity(lat, h):
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
        acc = self.accel_inv_error_matrix @ (acc - accel_bias)
        omega = self.gyro_inv_error_matrix @ (omega - self.gyro_bias)

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
            self.azimuth = atan(-self.alignment_omega_mean[0, 0] / self.alignment_omega_mean[1, 0])

            # Compute rotation matrix and the associated quaternion
            R_b2l = self.get_rotation_matrix(self.roll, self.pitch, self.azimuth)
            self.quat = self.matrix_to_normalized_quaternion(R_b2l)
            self.R_b2l = self.quaternion_to_matrix(self.quat)

            # Compute attitude errors after alignment
            accel_bias = self.accel_bias(g) if callable(self.accel_bias) else self.accel_bias
            self.post_alignment_roll_error = self.post_alignment_pitch_error = accel_bias / g
            omega_e_cos_phi = self.omega_e * cos(self.lat)
            self.post_alignment_azimuth_error = self.gyro_bias / omega_e_cos_phi + self.arw / (
                omega_e_cos_phi * sqrt(self.alignment_time)
            )

            # Flag alignment as complete
            self.alignment_complete = True

    def angular_velocity_compensation(self, N, M, omega, delta_t):
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

    def attitude_integration(self, d_theta_x, d_theta_y, d_theta_z):
        '''Update the quaternion and rotation matrices and compute roll, pitch, and azimuth'''

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
        self.pitch = atan(self.R_b2l[2, 1] / np.sqrt(self.R_b2l[0, 1] ** 2 + self.R_b2l[1, 1] ** 2))
        self.azimuth = atan(self.R_b2l[0, 1] / self.R_b2l[1, 1])

    def v_and_r_integration(self, acc, delta_t, g, N, M):
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

    def update_errors(self):
        '''
        BING CHAT:

        The error that has accumulated in attitude in time t due to the correlation time of a gyroscope
        can be calculated using the following equation:

        Error in attitude = (correlation time of gyroscope) * (angular rate of rotation) * sqrt(3t)

        where angular rate of rotation is the rate at which the gyroscope is rotating and t is the time
        for which the gyroscope has been used. This equation assumes that the gyroscope has a constant
        bias error. However, in reality, the bias error of a gyroscope is not constant and varies with
        time. Therefore, this equation provides only an approximate estimate of attitude error.

        The factor of sqrt(3t) comes from the fact that the error in attitude due to the correlation time
        of a gyroscope is a random walk process. The error grows with time and the rate of growth is
        proportional to the square root of time. The factor of sqrt(3) comes from the fact that the error
        in attitude due to a random walk process is proportional to the square root of the number of
        dimensions.
        '''

        if not self.alignment_complete:
            return
        return

        self.roll_error = self.post_alignment_roll_error + ...

    def process_measurement(self, measurement):
        '''Process a measurement from the IMU'''

        # In case we get a non-zero start time, shift the time steps back to the time since the first measurment
        if self.start_time is None:
            self.start_time = measurement[0]
        self.timestamp = measurement[0] - self.start_time
        omega = measurement[1:4].reshape((3, 1))
        acc = measurement[4:].reshape((3, 1))

        delta_t = self.timestamp - self.prev_time if self.prev_time is not None else 0
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
        d_theta_x, d_theta_y, d_theta_z = theta_blb.reshape(3)
        self.attitude_integration(d_theta_x, d_theta_y, d_theta_z)

        # Integrate to update position and LLF velocity
        self.v_and_r_integration(acc, delta_t, g, N, M)

        # Update errors
        self.update_errors()

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
                'isAligning',
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
            not self.alignment_complete,
        )
