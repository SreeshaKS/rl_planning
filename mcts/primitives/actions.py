import numpy as np 
from mcts.primitives.kinematics.dubins import Dubins

def rotation(theta):
    return np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


class PrimitiveActions:
    """
    Each primitive consists of a series of configurations.
    """

    def __init__(self, angle_range, step_count, animation_duration, velocity=1.0):
        assert len(angle_range) == 2
        self.horizon_range = angle_range
        self.count = step_count
        self.duration = animation_duration
        self.control_inputs = np.linspace(angle_range[0], angle_range[1], self.count)
        kinematics = Dubins(velocity)
        # Pre-computed primitive primitives
        primitives = []
        for action_idx in range(self.count):
            step = np.zeros(3)
            primitive = [step]
            for _ in range(animation_duration):
                step = kinematics.get_sweep(step, self.control_inputs[action_idx])
                primitive.append(step)
            primitives.append(primitive)
        self.primitives = np.asarray(primitives)
        assert self.primitives.shape[0] == self.count
        assert self.primitives.shape[1] == animation_duration + 1
        assert self.primitives.shape[2] == 3

    def get_sweep(self, steps, action_idx):
        assert steps.ndim == 1
        action = self.primitives[action_idx]

        # Rotation
        rotation_mat = rotation(steps[2])
        action = np.matmul(action, rotation_mat)
        action[:, 2] = (action[:, 2] + steps[2]) % (2 * np.pi)

        # Translation
        action[:, 0] += steps[0]
        action[:, 1] += steps[1]
        return action
