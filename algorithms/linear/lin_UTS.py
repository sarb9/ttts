import numpy as np

from algorithms.linear.lin_uniform import LinUniform
from algorithms.utils import add_chosen_arm


class LinUTS(LinUniform):
    @add_chosen_arm
    def choose(self, context):
        # sample from the baysian posterior
        features = np.array(self.features)
        # V = features.T @ features + self.reg * np.eye(self.d)
        theta_1 = np.random.multivariate_normal(self.thetas[-1], self.V_inv, 1)
        theta_2 = np.random.multivariate_normal(self.thetas[-1], self.V_inv, 1)
        theta = (theta_1 - theta_2)[0]
        arms = self.arms[context]
        arm = np.argmax(np.abs(arms @ theta))
        return arm
