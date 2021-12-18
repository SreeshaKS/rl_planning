from copy import copy
import numpy as np
import random
from mcts.primitives.actions import PrimitiveActions
class Bounds:
    def __init__(
        self,
        arena_bounds
        , n_iterations
        , animation_duration
        , rollout_bounds
    ):
        self.arena_bounds = arena_bounds
        self.n_iterations = n_iterations
        self.animation_duration = animation_duration
        self.rollout_bounds = rollout_bounds


class Node:
    def __init__(self,
                 step,
                 n_actions,
                 reward=0.0,
                 action=None,
                 parent=None):
        self.step = step
        self.pending_primitives = list(range(n_actions))
        self.reward = reward
        self.primitive = action
        self.parent = parent
        self.children = list()
        self.n_visits = 0

    def add_child(self, node):
        self.children.append(node)


class MCTS:
    def __init__(
            self,
            arena_bounds,
            horizon,
            velocity,
            num_actions,
            animation_duration,
            weight,
            max_iter,
            rollout_bounds,
            obstacle_penelty=-1.0,
    ):
        self.arena_bounds = arena_bounds
        self.n_primitives = num_actions
        # Number of primitives must be an odd number
        assert self.n_primitives % 2 != 0
        self.exploration_temp = weight
        self.n_iterations = max_iter
        self.rollout_bounds = rollout_bounds
        self.p_actions = PrimitiveActions(
            horizon,
            num_actions,
            animation_duration,
            velocity,
        )
        self.eps = 1e-6
        self.default_penalty = obstacle_penelty
        print(
            "********************************** MCTS initial PARAMS **************************************************",
            self.arena_bounds,
            horizon,
            velocity,
            self.n_primitives,
            animation_duration,
            self.exploration_temp,
            self.n_iterations,
            self.rollout_bounds
        )

    def run(self, pose, rewards, arena):
        # Save reward map and arena map
        self.rewards = rewards
        self.arena = arena
        assert self.arena.dtype == bool
        self.row_bounds = self.rewards.shape[0] - 1
        self.col_bounds = self.rewards.shape[1] - 1

        # Initialize root_node node
        self.root_node = Node(pose, self.n_primitives)
        # MCTS main loop
        for _ in range(self.n_iterations):
            # Selection
            explorable, is_valid = self._select(self.root_node)
            # This branch is blocked by obstacles.
            if not is_valid:
                continue
            # Expansion
            new_node = self._expand(explorable)
            # Simulation / rollout and backpropagation
            if new_node is None:  # No valid primitive available.
                self._backpropagation(explorable, self.default_penalty)
            else:
                self._backpropagation(new_node, self._simulate(new_node))

        if not self.root_node.children:
            x, y, angle = pose
            print(f"No valid primitive at [{x: .1f} {y: .1f} {angle: .1f}]")
            pose[2] += np.pi / 8
            print(f"Turn to [{pose[0]: .1f} {pose[1]: .1f} {pose[2]: .1f}]")
            return self.run(pose, rewards, arena)
        return self.best_action(self.root_node)

    def _select(self, node):
        # Select this node if it has some unvisited primitives
        if len(node.pending_primitives) != 0:
            return node, True
        # If all primitives have been visited but the children dict is empty
        elif not node.children:
            return node, False
        else:  # All primitives have been visited and the children dict is not empty
            exploitation_scores = [
                child.reward / child.n_visits for child in node.children
            ]
            exploration_scores = [
                np.sqrt(2.0 * np.log(node.n_visits) / child.n_visits)
                for child in node.children
            ]
            # Note that we rescaled the exploration exploration_temp according to
            # the maximum exploitation score.
            exploitation_scale = (np.max(exploitation_scores) + self.eps)
            explore_weight = exploitation_scale * self.exploration_temp
            ucb_metric = [
                exploit + explore_weight * explore for exploit, explore in zip(
                    exploitation_scores, exploration_scores)
            ]
            max_index = np.argmax(ucb_metric)
            return self._select(node.children[max_index])

    def boundary_check(self, action):
        if np.any(action[:, 0] <= self.arena_bounds[0]):
            return False
        if np.any(action[:, 0] >= self.arena_bounds[1]):
            return False
        if np.any(action[:, 1] <= self.arena_bounds[2]):
            return False
        if np.any(action[:, 1] >= self.arena_bounds[3]):
            return False
        return True

    def collision_check(self, coords):
        occupied = self.arena[coords[:, 0], coords[:, 1]]
        if np.any(occupied):
            return False
        return True

    def _expand(self, parent):
        # Randomly select an primitive from the available primitives
        action_idx = parent.pending_primitives[0]
        del parent.pending_primitives[0]
        action = self.p_actions.get_sweep(parent.step, action_idx)

        # Check whether this primitive is valid
        coords = self.__mapping(action[:, :2], self.arena_bounds, self.row_bounds, self.col_bounds)
        is_inside_boundary = self.boundary_check(action)
        is_outside_obstacle = self.collision_check(coords)
        is_valid_action = is_inside_boundary and is_outside_obstacle

        # Create the child node and attach it to its parent
        child = None
        if is_valid_action:
            pose = action[-1]
            rewards = self.rewards[coords[:, 0], coords[:, 1]]
            reward = np.sum(rewards, axis=0)
            child = Node(pose, self.n_primitives, reward, action, parent)
            parent.add_child(child)
        return child

    def _simulate(self, node):
        # Default policy is moving forward
        action_idx = self.n_primitives // 2
        step = copy(node.step)
        steps = []
        for _ in range(self.rollout_bounds):
            action = self.p_actions.get_sweep(step, action_idx)
            step = action[-1]
            steps.append(action)
        steps = np.vstack(steps)
        coords = self.__mapping(steps[:, :2], self.arena_bounds, self.row_bounds, self.col_bounds)
        rewards = self.rewards[coords[:, 0], coords[:, 1]]
        reward = np.sum(rewards, axis=0)
        average_reward = reward / self.rollout_bounds
        return average_reward

    def _backpropagation(self, node, reward):
        node.reward += reward
        node.n_visits += 1
        if node.parent is not None:
            self._backpropagation(node.parent, reward)

    def best_action(self, node):
        assert len(node.pending_primitives) == 0
        if not node.children:
            raise ValueError(
                "No valid primitive in current steps!\n"
                "You might need to implement some 'turn around' engineering "
                "tricks to solve this problem.")

        ## Uncomment this block to use n_visits instead of expected reward
        #  n_visits = [child.n_visits for child in node.children]
        #  idx = np.argmax(n_visits)
        #  best_child = node.children[idx]
        #  print("Number of visits: ", n_visits)

        values = [child.reward / child.n_visits for child in node.children]

        return node.children[np.argmax(values)].primitive

    def get_paths(self, max_depth=None):
        
        if not self.root_node.children:
            return None
        steps = []
        node = copy(self.root_node)
        depth = 0
        while node.children:
            values = [
                child.reward / child.n_visits for child in node.children
            ]
            idx = np.argmax(values)
            winner = node.children[idx]
            steps.append(winner.primitive)
            node = winner
            depth += 1
            if max_depth is not None and depth == max_depth:
                break
        return np.vstack(steps)

    def dfs(self, node, steps):
        steps.append(node.primitive)
        if node.children:
            for child in node.children:
                steps = self.dfs(child, steps)
        return steps

    def get_progression(self):
        steps = []
        for child in self.root_node.children:
            steps = self.dfs(child, steps)
        return np.vstack(steps)


    def __mapping(self, xy, bounds, max_row, max_col):
        # x -> j
        j = (xy[:, 0] - bounds[0]) / (bounds[1] - bounds[0]) * max_col
        j = np.round(j, decimals=6)
        j[j < 0] = 0
        j[j > max_col] = max_col
        # y -> i
        i = (xy[:, 1] - bounds[2]) / (bounds[3] - bounds[2]) * max_row
        i = np.round(i, decimals=6)
        i[i < 0] = 0
        i[i > max_row] = max_row
        
        return np.vstack([i.ravel(), j.ravel()]).T.astype(np.int)