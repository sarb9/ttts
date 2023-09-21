from algorithms.utils import EcoLog
from algorithms.uniform import Uniform
import numpy as np
from algorithms.utils import add_chosen_arm


class ECOLOG(Uniform):
    def __init__(
        self,
        arms,
        true_param_norm_ub=None,
        failure_level=0.05,
        **kwargs,
    ):
        super().__init__(arms, **kwargs)
        assert (
            true_param_norm_ub is not None
        ), "true_param_norm_ub must be specified for Confidence algorithm"
        self.true_param_norm_ub = true_param_norm_ub
        self.failure_level = failure_level
        # Assert all arms have norm bounded by true_param_norm_ub
        self.arm_norm_ub = 1
        assert np.all(
            np.linalg.norm(self.arms, axis=2) <= self.true_param_norm_ub
        ), "All arms must have norm bounded by true_param_norm_ub"
        self.ecolog = EcoLog(
            param_norm_ub=self.true_param_norm_ub,
            arm_norm_ub=self.arm_norm_ub,
            dim=self.d,
            failure_level=self.failure_level,
        )
        self.armpairs = self.create_armpairs()

    def create_armpairs(self):
        # We need to create a set of arms based on pairing "different" arms together
        n_pairs = self.n_arms**2 - self.n_arms
        armpairs = np.zeros((self.n_contexts, n_pairs, self.d))
        for i in range(self.n_contexts):
            for j in range(self.n_arms):
                sub_one = 0
                for k in range(self.n_arms):
                    if j == k:
                        sub_one = 1
                        continue
                    armpairs[i, j * (self.n_arms - 1) + k - sub_one] = (
                        self.arms[i, j] - self.arms[i, k]
                    )
        return armpairs

    @add_chosen_arm
    def choose(self, context):
        # Record the theta
        self.thetas.append(self.ecolog.theta)
        # Choose arms according to the ECOLog algorithm
        active_arms = self.armpairs[context]
        arm_pair_index = self.ecolog.pull(active_arms)
        # We need to convert the arm_pair_index to the index of the original arms
        arm1 = arm_pair_index // (self.n_arms - 1)
        arm2 = arm_pair_index % (self.n_arms - 1)
        if arm2 >= arm1:
            arm2 += 1
        return arm1, arm2

    def learn(self, context, arm, reward):
        # Update the ECOLog algorithm
        arm1, arm2 = arm
        self.ecolog.learn(
            arm=self.armpairs[context, arm1 * (self.n_arms - 1) + arm2 - 1],
            reward=reward,
        )

    def get_policy(self, checkpoint=None):
        theta_hat = None
        if checkpoint is None:
            theta_hat = self.thetas[-1]
        else:
            theta_hat = self.thetas[checkpoint]

        def policy(context):
            # Returns the arm with the highest mean reward
            arms = self.arms[context]
            return np.argmax(np.dot(arms, theta_hat))

        return policy
