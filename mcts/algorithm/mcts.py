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


class MCTS:
    def __init__(
            self,
            algorithm_bounds,
        ):
        self.algorithm_bounds = algorithm_bounds
    
    def run(self, step, rewards, arena):
        pass

    def _select(self):
        pass
    
    def _expand(self):
        pass
    
    def _simulate(self):
        pass
    
    def _backpropagate(self):
        pass

    def rollout(self):
        pass


    