from copy import copy
import numpy as np
import random

class Bounds:
    def __init__(
        self,
        arena_bounds
        , n_iterations_bound
        , rollout_bounds
        , num_primitive_actions
        , animation_duration
    ):
        self.arena_bounds = arena_bounds
        self.n_iterations_bound = n_iterations_bound
        self.rollout_bounds = rollout_bounds
        self.num_primitive_actions = num_primitive_actions
        self.animation_duration = animation_duration

class Node:
    def __init__(self,
                 step,
                 n_actions,
                 reward=0.0,
                 primitve=None,
                 parent=None):
        self.step = step
        self.pending_actions = list(range(n_actions))
        self.reward = reward
        self.primitve = primitve
        self.parent = parent
        self.children = list()
        self.n_visits = 0

    def add_child(self, node):
        self.children.append(node)


class MCTS:
    def __init__(
            self,
            algorithm_bounds,
            default_penalty,
        ):
        self.default_penalty = default_penalty
        self.algorithm_bounds = algorithm_bounds
    
    def run(self, step, rewards, arena):
        self.rewards = rewards
        self.arena = arena

        self.root_node = Node(step, self.bounds.num_primitive_actions)

        for _ in range(self.bounds.num_primitive_actions):
            expandable, valid = self._select(self.root_node)
            if not valid:
                continue

            new_child = self._simulate(expandable)
            if not new_child:
                self._backpropagate(expandable, self.self.default_penalty)
            else:
                reward = self._expand(new_child)
                self._backpropagate(new_child, reward)

            if len(self.root.children) == 0:
                x, y, angle = step
                print(f"No valid action at [{x: .1f} {y: .1f} {angle: .1f}]")
                step[2] += np.pi / 8
                return self.run(step, rewards, arena)
        
        if len(self.root_node.children) == 0:
            return None
        
        max_index = np.argmax(
            [node.reward / node.n_visits for node in self.root_node.children]
        )
        
        return self.root_node.children[max_index]

    def _select(self, node):
        # Select this node if it has some pending actions
        if len(node.pending_actions) != 0:
            return node, True
        # If all actions have been visited but empty children
        elif not node.children:
            return node, False
        else:
            # all actions are executed but more children to expand
            exploitation_metrices = []
            for node in node.children:
                exploitation_metrices.append(
                    node.reward / node.n_visits
                )
            eploration_metrices = []
            for node in node.children:
                eploration_metrices.append(
                    np.sqrt(
                        2.0 * np.log(node.n_visits) /
                        node.n_visits
                    )
                )
            cross_array = zip(exploitation_metrices, eploration_metrices)
            # get UCB metric for all children
            ucb_metric = []
            for exploit, explore in cross_array:
                ucb_metric.append(
                    exploit + explore
                )
            max_index = np.argmax(ucb_metric)
            selected_child, valid = self._select(node.children[max_index])
            return selected_child, valid
            
    
    def _simulate(self, node):
        random_primitive = random.randrange(0, len(node.pending_actions), 1)
        primitive_id = node.pending_actions[random_primitive]
        del node.pending_actions[random_primitive]

        primitive = self.get_sweep(node.step, primitive_id)
        coords = self.__mapping(primitive[:, :2], self.bounds, self.max_row, self.max_col)

        valid_move = self._is_valid_move(primitive, coords)
        # update validity

        child = None
        if valid_move:
            step = primitive[-1]
            rewards = self.rewards[coords[:, 0], coords[:, 1]]
            reward = np.sum(rewards, axis=0)
            node.add_child(Node(
                step,
                self.n_actions,
                reward,
                primitive,
                node
            ))
        return child

    def _expand(self, node):
        primitive_idx = self.n_actions // 2
        step = copy(node.state)
        moves = []
        for _ in range(self.max_rollout):
            primitive = self.get_sweep(node.step, primitive_idx)
            step = primitive[-1]
            moves.append(step)
        moves = np.vstack(moves)
        coords = self.__mapping(primitive[:, :2], self.bounds, self.max_row, self.max_col)
        rewards = self.rewards[coords[:, 0], coords[:, 1]]
        
        return np.sum(rewards, axis=0) / self.max_rollout

    
    def _backpropagate(self, node, reward):
        node.reward += reward
        node.n_visits += 1
        if node.parent is not None:
            self._backpropagate(node.parent, reward)

    
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
        # stack
        ij = np.vstack([i.ravel(), j.ravel()]).T.astype(np.int)
        return ij



    def _is_valid_move(self, primitive, coords):
        # Check whether this action is valid
        is_inside_boundary = self.boundary_check(primitive)
        is_outside_obstacle = self.collision_check(coords)
        return is_inside_boundary and is_outside_obstacle
    
    def _boundary_check(self, primitive):
        if np.any(primitive[:, 0] <= self.boundary[0]) \
            or np.any(primitive[:, 0] >= self.boundary[1]) \
            or np.any(primitive[:, 1] <= self.boundary[2])\
            or np.any(primitive[:, 1] >= self.boundary[3]):
            return False
        return True

    def collision_check(coords):
        pass

    