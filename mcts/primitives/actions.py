import numpy as np 
from kinematics import Dubins

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
    Compute trajectory and movements of the robot
    Each primitive action consists of a series of configurations
    """
    def __init__(self, horizon_range, step_count, animation_duration, velocity):
        self.horizon_range = horizon_range
        self.animation_duration = animation_duration
        self.primitive_steps = np.linspace(horizon_range[0], horizon_range[1], step_count)
        kinematics = Dubins(velocity)
        primitive_steps = []
        for step_index in range(step_count):
            step = np.zeros(3)
            primitive = [step]
            for _ in range(animation_duration):
                primitive.append(kinematics.get_sweep)
            primitive_steps.append(primitive)
        self.primitive_steps = np.array(primitive_steps)
        
        assert self.primitive_steps.shape == (step_count, animation_duration+1, 3)
    
    def get_primitive(self, step, step_primitive_index):
        primitive = self.primitive_steps[step_primitive_index]

        rotation_matrix = rotation(step[2])
        primitive = np.matmul(primitive, rotation_matrix)
        primitive[:,2] = (primitive[:,2] + step[2]) % (2 * np.pi)

        # Translation
        primitive[:, 0] += step[0]
        primitive[:, 1] += step[1]
        return primitive

