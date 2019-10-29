import math


class Viewport:
    """
        Used to create viewport necessary for viewport based psnr calculation
    """

    def __init__(self, width, height, fov_x):
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


class Vector3:
    """
        Defines a 3D vector
    """

    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

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
