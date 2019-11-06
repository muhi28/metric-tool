import math

import numpy as np


def _almost_equal(a, b, delta):
    return math.fabs(a - b) < delta


def get_pitch(vec3):
    angle_v = Vector3(vec3.x, vec3.y, 0)

    degree = math.degrees(angle_v.angle(vec3))
    degree = degree * -1.0 if vec3.z > 0 else degree

    return degree


def get_yaw(vector):
    angle_vec = Vector3(vector.x, vector.y, vector.z)
    x_axis = Vector3(1, 0, 0)

    angle_vec.x = 0 if _almost_equal(vector.x, 0, 1e-10) else vector.x
    angle_vec.y = 0 if _almost_equal(vector.y, 0, 1e-10) else vector.y

    degree = math.degrees(angle_vec.angle(x_axis))
    degree = 360 - degree if vector.y < 0 else degree

    return degree


def _get_rotation_matrix(alpha, beta, gamma):
    a = math.radians(alpha)
    b = math.radians(beta)
    g = math.radians(gamma)

    # perform rotations around x, y and z axis
    rz = Matrix3(math.cos(a), -math.sin(a), 0,
                 math.sin(a), math.cos(a), 0,
                 0, 0, 1)

    ry = Matrix3(math.cos(b), 0, math.sin(b),
                 0, 1, 0,
                 -math.sin(b), 0, math.cos(b))

    rx = Matrix3(1, 0, 0,
                 0, math.cos(g), -math.sin(g),
                 0, math.sin(g), math.cos(g))

    rz.multiply(ry)
    rz.multiply(rx)

    return rz


class Viewport:
    """
        Used to create viewport necessary for viewport based psnr calculation
    """

    def __init__(self, width=0, height=0, fov_x=0):
        self._width = width
        self._height = height
        self._fov_x = fov_x

    def get_width(self):
        """
            Get viewport width
        :return: vp width
        """
        return self._width

    def set_width(self, width):
        """
            Set viewport width
        :param width: new viewport width
        :return: none
        """
        self._width = width

    def get_height(self):
        """
            Get viewport height
        :return: vp height
        """
        return self._height

    def set_height(self, height):
        """
            Set viewport height
        :param height: new vp height
        :return: none
        """
        self._height = height

    def set_fov_x(self, fov):
        self._fov_x = fov

    def get_fov_x(self):
        """
            Get field of view X for viewport
        :return: field of view in x
        """
        return self._fov_x

    def get_fov_y(self):
        """
            Get field of view X for viewport
        :return:
        """
        return 2.0 * math.atan(self._height / (2 * self.get_focal_len()))

    def get_focal_len(self):
        """
            Calculate the focal length
        :return: focal length in pixel
        """
        _fov_x_half_rad = float(math.radians(self._fov_x / 2.0))
        result = float(self._width / (2.0 * math.tan(_fov_x_half_rad)))
        return result

    def _get_transform_3d_2_2d(self):
        res = Matrix3(self.get_focal_len(), 0, 0,
                      0, self.get_focal_len(), 0,
                      0, 0, 0)

        return res

    def get_transform_2d_2_3d(self):
        mat3 = self._get_transform_3d_2_2d()

        return mat3.inverse()

    def get_spherical_coords(self, viewport_coords, viewing_direction):
        yaw = get_yaw(viewing_direction)
        pitch = get_pitch(viewing_direction)

        rotation = _get_rotation_matrix(yaw, pitch, 0)

        adapt = Vector3(viewport_coords.x, viewport_coords.y, 1)

        tmp = self.get_transform_2d_2_3d().transform(adapt)

        res_vec = Vector3(tmp.z, tmp.x, tmp.y)

        res_vec = rotation.transform(res_vec)
        res_vec.normalize()

        return res_vec


class Matrix3:

    def __init__(self, m00=0.0, m01=0.0, m02=0.0, m10=0.0, m11=0.0, m12=0.0, m20=0.0, m21=0.0, m22=0.0):
        self.matrix = np.asmatrix(np.zeros((3, 3), np.float))

        self.matrix[0, 0] = m00
        self.matrix[0, 1] = m01
        self.matrix[0, 2] = m02
        self.matrix[1, 0] = m10
        self.matrix[1, 1] = m11
        self.matrix[1, 2] = m12
        self.matrix[2, 0] = m20
        self.matrix[2, 1] = m21
        self.matrix[2, 2] = m22

    def multiply(self, m):
        self.matrix.dot(m)

    def inverse(self):
        return self.matrix.getI()

    def transform(self, vec3):
        return Vector3(self.matrix[0, 0] * vec3.x + self.matrix[0, 1] * vec3.y + self.matrix[0, 2] * vec3.z,
                       self.matrix[1, 0] * vec3.x + self.matrix[1, 1] * vec3.y + self.matrix[1, 2] * vec3.z,
                       self.matrix[2, 0] * vec3.x + self.matrix[2, 1] * vec3.y + self.matrix[2, 2] * vec3.z)


class Vector3:
    """
        Defines a 3D vector
    """

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def get_x(self):
        return self.x

    def set_x(self, x):
        self.x = x

    def get_y(self):
        return self.y

    def set_y(self, y):
        self.y = y

    def get_z(self):
        return self.z

    def set_z(self, z):
        self.z = z

    def angle(self, vec):
        _mag = self.x * self.x + self.y * self.y + self.z * self.z
        _vmag = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z

        _dot_val = self.x * vec.x + self.y * vec.y + self.z * vec.z
        return math.acos(_dot_val / math.sqrt(_mag * _vmag))

    def normalize(self):
        np_arr = np.array([self.x, self.y, self.z])

        np_arr = (np_arr / np.linalg.norm(np_arr, ord=2, axis=1, keepdims=True))

        self.x = np_arr[0][0]
        self.y = np_arr[0][1]
        self.z = np_arr[0][2]
