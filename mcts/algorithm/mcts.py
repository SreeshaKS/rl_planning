import numpy as np


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
        pass

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
            
    
    def _expand(self, node):
        pass
    
    def _simulate(self):
        pass
    
    def _backpropagate(self, node, reward):
        node.reward += reward
        node.n_visits += 1
        if node.parent is not None:
            self._backpropagate(node.parent, reward)

    def rollout(self, node):
        pass


    