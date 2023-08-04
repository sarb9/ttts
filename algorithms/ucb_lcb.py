from algorithms.ecolog import ECOLOG
import numpy as np


class UCBLCB(ECOLOG):
    def get_arms(self, context):
        # Record the theta
        self.thetas.append(self.ecolog.theta)
        # Choose arms according to our strategy
        max_ucb = -np.inf
        max_ucb_arm = None
        for i, arm in enumerate(self.arms[context]):
            ucb = self.ecolog.compute_optimistic_reward(arm)
            if ucb > max_ucb:
                max_ucb = ucb
                max_ucb_arm = i
        lcb = self.ecolog.compute_pessimistic_reward(self.arms[context][max_ucb_arm])
        arm2 = None
        arm2_delta = -np.inf
        for i, arm in enumerate(self.arms[context]):
            if i == max_ucb_arm:
                continue
            ucb_arm = self.ecolog.compute_optimistic_reward(arm)
            if ucb_arm < lcb:
                continue
            delta = np.abs(
                np.dot(self.ecolog.theta, arm - self.arms[context][max_ucb_arm])
            )
            if delta > arm2_delta:
                arm2_delta = delta
                arm2 = i
        if arm2 is None:
            # pick some arm at random that is not the max_ucb_arm
            arm2 = np.random.choice(
                [i for i in range(len(self.arms[context])) if i != max_ucb_arm]
            )
        return max_ucb_arm, arm2
