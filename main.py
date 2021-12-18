from image_to_grid import convertImageToGrid
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mcts.algorithm.mcts import MCTS
from mcts.algorithm.mcts import Bounds
from mcts.primitives.actions import PrimitiveActions
import re

plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "gray_r"

# Arena edges
arena_bounds = [0, 100, 0, 100]
print("Environment arena_bounds: ", arena_bounds)

# Robot's initial configuration
step = np.array([50, 50, -100 ])
print(f"Robot's state: [x={step[0]}, y={step[1]}, yaw={step[2]}]")

arena_grid = convertImageToGrid('test.png')
arena_grid = (arena_grid - arena_grid.min()) / arena_grid.max()


def normalize(grid):
    row_sums = grid.sum(axis=1)
    return grid / row_sums[:, np.newaxis]

def coinFlip(p):
    return np.random.binomial(1,p)

reward_density = 0.7

# Occupancy grid map is now a boolean matrix with 1 indicating occupied.
arena_grid = arena_grid < 0.5
assert arena_grid.dtype == bool

# Create an artificial reward map.
reward_matrix = np.empty(arena_grid.shape)
reward_map = reward_matrix.astype(np.float32)
for i in range(reward_matrix.shape[0]):
    for j in range(reward_matrix.shape[1]):
        
        y = coinFlip(reward_density)
        x = coinFlip(reward_density)

        r_x = i
        r_y = j
        # r_x = reward_matrix.shape[1] - i
        # r_y = reward_matrix.shape[0] - j

        if arena_grid[i, j] != 1:
            reward_matrix[i, j] = r_x * y  + x * r_y
        else:
            reward_matrix[i, j] = 0

reward_map[reward_matrix.shape[0]-1][reward_matrix.shape[1]-1] = 1
reward_map = (reward_matrix - reward_matrix.min()) / reward_matrix.max()



print("Shape of arena: ", arena_grid.shape)
print("Shape of reward: ", reward_matrix.shape)

# Planning
horizon = [-0.1, 0.1]  # Steering angle range
velocity = 1.0
n_actions = 5  # Number of primitive primitives
animation_duration = 10  # Number of contiguration points ([x, y, theta]) per primitive
exploration_temp = 0.9  # Exploration exploration_temp. Larger weights will lead to fatter trees.
max_iter = 1000  # Maximum number of tree search iterations
expansion_bounds = 5  # Maximum number of rollouts or simulations
mcts = MCTS(
    arena_bounds,
    horizon,
    velocity,
    n_actions,
    animation_duration,
    exploration_temp,
    max_iter,
    expansion_bounds,
)
best_action = mcts.run(step, reward_map, arena_grid)
progressions = mcts.get_progression()
primitives = mcts.get_paths()

print("Red arrows represent the current best primitive.")
print("Blue arrows show the best primitives")
print("Yellow dots demonstrate the searching tree.")

"""
Drawing phase space trajectories with arrows in matplotlib

Reference - https://stackoverflow.com/questions/36607742/drawing-phase-space-trajectories-with-arrows-in-matplotlib
https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.quiver.html
https://problemsolvingwithpython.com/06-Plotting-with-Matplotlib/06.15-Quiver-and-Stream-Plots/
"""
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

ogm = axes[0].imshow(arena_grid, extent=arena_bounds)
axes[0].quiver(
    progressions[:, 0],
    progressions[:, 1],
    np.cos(progressions[:, 2]),
    np.sin(progressions[:, 2]),
    color="y",
    alpha=0.5,
)
axes[0].quiver(
    best_action[:, 0],
    best_action[:, 1],
    np.cos(best_action[:, 2]),
    np.sin(best_action[:, 2]),
    color="r",
    alpha=0.5,
)
axes[0].quiver(
    primitives[:, 0],
    primitives[:, 1],
    np.cos(primitives[:, 2]),
    np.sin(primitives[:, 2]),
    color="b",
    alpha=0.1,
)
divider = make_axes_locatable(axes[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(ogm, cax=cax)
axes[0].set_title("Arena")

rm = axes[1].imshow(reward_map, extent=arena_bounds)
axes[1].quiver(
    progressions[:, 0],
    progressions[:, 1],
    np.cos(progressions[:, 2]),
    np.sin(progressions[:, 2]),
    color="y",
    alpha=0.5,
)
axes[1].quiver(
    best_action[:, 0],
    best_action[:, 1],
    np.cos(best_action[:, 2]),
    np.sin(best_action[:, 2]),
    color="b",
    alpha=0.5,
)
axes[1].quiver(
    primitives[:, 0],
    primitives[:, 1],
    np.cos(primitives[:, 2]),
    np.sin(primitives[:, 2]),
    color="r",
    alpha=0.1,
)
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(rm, cax=cax)
axes[1].set_title("reward density")

fig.tight_layout()
plt.savefig("output.png")
plt.show()

