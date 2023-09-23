import numpy as np

from algorithms.linear.lin_uniform import LinUniform
from algorithms.utils import add_chosen_arm


class LinTS(LinUniform):
    @add_chosen_arm
    def choose(self, context):
        # sample from the baysian posterior
        features = np.array(self.features)
        # V = (features.T @ features + self.reg * np.eye(self.d)) / len(self.thetas)
        theta = np.random.multivariate_normal(self.thetas[-1], self.V_inv, 1)[0]
        # print("sampled theta:", theta, "theta:", self.thetas[-1], "V: ", V)
        arms = self.arms[context]
        arm = np.argmax(np.abs(arms @ theta))
        return arm
