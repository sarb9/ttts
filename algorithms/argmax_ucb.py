from algorithms.ecolog import ECOLOG
import numpy as np


class ArgmaxUCB(ECOLOG):
    def get_arms(self, context):
        # Record the theta
        self.thetas.append(self.ecolog.theta)
        # Choose arms according to our strategy
        max_value = -np.inf
        max_value_arm = None
        for i, arm in enumerate(self.arms[context]):
            value = np.dot(self.ecolog.theta, arm)
            if value > max_value:
                max_value = value
                max_value_arm = i
        max_ucb = -np.inf
        max_ucb_arm = None
        for i, arm in enumerate(self.arms[context]):
            if i == max_value_arm:
                continue
            ucb_arm = self.ecolog.compute_optimistic_reward(arm)
            if ucb_arm > max_ucb:
                max_ucb = ucb_arm
                max_ucb_arm = i
        return max_value_arm, max_ucb_arm
