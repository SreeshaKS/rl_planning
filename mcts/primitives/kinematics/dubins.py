"""
Dubins car dynamics.
Primary reference - https://gieseanw.wordpress.com/2012/10/21/a-comprehensive-step-by-step-tutorial-to-computing-dubins-paths
Library reference - https://pypi.org/project/dubins
Secondary reference - http://planning.cs.uiuc.edu/node821.html
"""
import numpy as np


class Dubins:
    """
    Dubins car dynamics with fixed velocity.

    :param velocity: forward velocity, defaults to 1.0
    :type velocity: float, optional
    """

    def __init__(self, velocity=1.0):
        self.velocity = velocity

    @property
    def velocity(self):
        return self.__velocity

    @velocity.setter
    def velocity(self, v):
        self.__velocity = v

    def get_sweep(self, step, angle):
        """
        Angular Sweep of the robot given its current step and the executed turning angle.
        Implementation reference - https://github.com/EwingKang/Dubins-RRT-for-MATLAB
        :param step: position and orientation [$x_1$, $x_2$, $\theta$]
        :type step: numpy.ndarray
        :param angle: steering angle
        :type angle: float
        """
        assert -np.pi <= angle <= np.pi
        new_step = step.copy()
        new_step[0] = self.velocity * np.cos(step[2]) + step[0]
        new_step[1] = self.velocity * np.sin(step[2]) + step[1]
        new_step[2] = (step[2] + angle) % (2 * np.pi)
        return new_step
