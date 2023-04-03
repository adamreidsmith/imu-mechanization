from math import copysign, sqrt, sin, cos, tan, asin, atan, pi


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
        h0,  # initial height
        lat0,  # initial latitude
        long0,  # initial longitude
        accel_bias=0,  # accel bias
        gyro_bias=0,  # gyro bias
        accel_sf=((0, 0, 0), (0, 0, 0), (0, 0, 0)),  # accel scale factor matrix
        gyro_sf=((0, 0, 0), (0, 0, 0), (0, 0, 0)),  # gyro scale factor matrix
        accel_no=((0, 0, 0), (0, 0, 0), (0, 0, 0)),  # accel non-orthogonality
        gyro_no=((0, 0, 0), (0, 0, 0), (0, 0, 0)),  # gyro non-orthogonality
        vrw=0,  # velocity random walk
        arw=0,  # angle random walk
        accel_corr_time=0,  # accel correlation time
        gyro_corr_time=0,  # gyro correlation time
        accel_bias_instability=0,  # accel bias instability
        gyro_bias_instability=0,  # gyro bias instability
        alignment_time=0,  # alignment time in seconds
    ):
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
        self.accel_inv_error_matrix = self.matinv(self.matsum(self.matsum(self.i3, accel_sf), accel_no))
        self.gyro_inv_error_matrix = self.matinv(self.matsum(self.matsum(self.i3, gyro_sf), gyro_no))
        self.quat = None  # rotation quaternion; inititalized when self.align is called
        self.R_b2l = None  # rotation matrix; inititalized when self.align is called
        self.v_llf = [0, 0, 0]  # initial ENU velocities
        self.prev_delta_v_llf = [0, 0, 0]  # change in v in LLF at previous t
        self.prev_time = 0  # previous time, used to determine delta_t
        self.roll = self.pitch = self.azimuth = 0  # initialize to store orientation
        self.timestamp = 0  # store the current timestamp

        # Alignment specific attributes
        self.alignment_time = alignment_time
        self.alignment_complete = False  # flag to determine if alignment is completed
        self.alignment_acc_mean = [0, 0, 0]  # running mean for accel alignment
        self.alignment_omega_mean = [0, 0, 0]  # running mean for gyro alignment
        self.alignment_it = 0  # couter to keep track of alignment iterations
        self.post_alignment_roll_error = -1  # roll error after alignment
        self.post_alignment_pitch_error = -1  # pitch error after alignment
        self.post_alignment_azimuth_error = -1  # azimuth error after alignment

        # Errors
        self.roll_error = 0
        self.pitch_error = 0
        self.azimuth_error = 0

    @staticmethod
    def matinv(m):
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
        return [
            [m1[0][0] + m2[0][0], m1[0][1] + m2[0][1], m1[0][2] + m2[0][2]],
            [m1[1][0] + m2[1][0], m1[1][1] + m2[1][1], m1[1][2] + m2[1][2]],
            [m1[2][0] + m2[2][0], m1[2][1] + m2[2][1], m1[2][2] + m2[2][2]],
        ]

    @staticmethod
    def matvec(m, v):
        return [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]

    @staticmethod
    def matvec4(m, v):
        return [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
            m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3],
        ]

    @staticmethod
    def T3(m):
        a, b, c = m[0]
        d, e, f = m[1]
        g, h, i = m[2]
        return [[a, d, g], [b, e, h], [c, f, i]]

    @staticmethod
    def get_rotation_matrix(roll, pitch, azimuth):
        '''Compute the rotation matrix to rotate the body frame to the LLF from the Euler angles'''

        cosr = cos(roll)
        cosp = cos(pitch)
        cosa = cos(azimuth)
        sinr = sin(roll)
        sinp = sin(pitch)
        sina = sin(azimuth)
        cosacosr = cosa * cosr
        sinasinr = sina * sinr
        sinacosr = sina * cosr
        cosasinr = cosa * sinr
        return [
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
            self.azimuth = atan(-self.alignment_omega_mean[0] / self.alignment_omega_mean[1])

            # Compute rotation matrix and the associated quaternion
            R_b2l = self.get_rotation_matrix(self.roll, self.pitch, self.azimuth)
            self.quat = self.matrix_to_quaternion(R_b2l)
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

        self.timestamp = measurement[0]
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
                *self.v_llf,
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
