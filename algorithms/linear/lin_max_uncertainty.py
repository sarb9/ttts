import numpy as np

from algorithms.linear.lin_uniform import LinUniform
from algorithms.utils import add_chosen_arm


class LinMaxUncertainty(LinUniform):
    @add_chosen_arm
    def choose(self, context):
        arms = self.arms[context]
        uncertainty = np.zeros((self.n_arms,))
        for arm in range(self.n_arms):
            uncertainty[arm] = np.sqrt(arms[arm] @ self.V_inv @ arms[arm])
        arm = np.argmax(uncertainty)
        return arm
