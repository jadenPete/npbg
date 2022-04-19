""" Camera class for 3D viewpoint manipulation. """

import functools
from glumpy.ext import glfw
import math
import numpy as np
import scipy

class PositionalCamera:
    """ An alternative to the old trackball class for first person video game-like navigation. """

    def __init__(self, pose, size, speed):
        """
        Args:
            pose (4, 4): The initial view matrix.
            size (2,): The initial window size.
            speed (float): The movement speed.
        """

        self._pose = pose
        self._rotation_base = pose[:3, :3].copy()
        self._size = size
        self._speed = speed
        self._velocity = np.array([0.0, 0.0, 0.0])
        self._velocity_delta = {
            glfw.GLFW_KEY_W: np.array([0.0, 0.0, -1.0]),
            glfw.GLFW_KEY_A: np.array([-1.0, 0.0, 0.0]),
            glfw.GLFW_KEY_S: np.array([0.0, 0.0, 1.0]),
            glfw.GLFW_KEY_D: np.array([1.0, 0.0, 0.0]),
            glfw.GLFW_KEY_SPACE: np.array([0.0, 1.0, 0.0]),
            glfw.GLFW_KEY_LEFT_SHIFT: np.array([0.0, -1.0, 0.0])
        }

        self._pressed = {key: False for key in self._velocity_delta}

    def _replace_shift(fn):
        """ A decorator to convert an unknown key code to the left shift key. """

        @functools.wraps(fn)
        def wrapper(self, key):
            if key == glfw.GLFW_KEY_UNKNOWN:
                key = glfw.GLFW_KEY_LEFT_SHIFT

            return fn(self, key)

        return wrapper

    def motion(self, point):
        """ Registers mouse movement to a point on the screen.

        Args:
            point (2,): Said screen coordinate.
        """

        yaw = math.pi - math.pi * 2 * point[0] / self._size[0]
        pitch = math.pi / 2 - math.pi * point[1] / self._size[1]

        yaw_mat = scipy.spatial.transform.Rotation.from_rotvec([0, yaw, 0]).as_matrix()
        pitch_mat = scipy.spatial.transform.Rotation.from_rotvec([pitch, 0, 0]).as_matrix()

        self._pose[:3, :3] = self._rotation_base @ yaw_mat @ pitch_mat

    def pose(self, dt):
        """ Updates and returns the view matrix.

        Args:
            dt (float): The time elapsed since the last invocation.

        Returns (4, 4):
            Said updated view matrix.
        """

        self._pose[:3, 3] += self._pose[:3, :3] @ (self._velocity * dt)

        return self._pose

    @_replace_shift
    def press(self, key):
        """ Registers a key press.

        Args:
            key (int): The key code.

        Returns:
            True if it is relevant to movement; otherwise, False.
        """

        if key not in self._pressed:
            return False

        if not self._pressed[key]:
            self._velocity += self._velocity_delta[key] * self._speed

            self._pressed[key] = True

        return True

    @_replace_shift
    def release(self, key):
        """ Registers a key release.

        Args:
            key (int): The key code.

        Returns:
            True if it is relevant to movement; otherwise, False.
        """

        if key not in self._pressed:
            return False

        if self._pressed[key]:
            self._velocity -= self._velocity_delta[key] * self._speed

            self._pressed[key] = False

        return True

    def resize(self, size):
        """ Registers a window resize.

        Args:
            size (2,): The new window size.
        """

        self._size = size
