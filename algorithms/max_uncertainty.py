from algorithms.ecolog import ECOLOG
import numpy as np
from algorithms.utils import add_chosen_arm
from utils.utils import weighted_norm


class MaxUncertainty(ECOLOG):
    @add_chosen_arm
    def choose(self, context):
        # Record the theta
        self.thetas.append(self.ecolog.theta)
        active_arms = self.armpairs[context]
        max_uncertainty = -np.inf
        max_arm_pair_index = None
        for i, arm in enumerate(active_arms):
            uncertainty = weighted_norm(arm, self.ecolog.vtilde_matrix_inv)
            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                max_arm_pair_index = i

        arm1 = max_arm_pair_index // (self.n_arms - 1)
        arm2 = max_arm_pair_index % (self.n_arms - 1)

        if arm2 >= arm1:
            arm2 += 1

        # if min(arm1, arm2) == 1 and max(arm1, arm2) == 2:
        #     print(active_arms[max_arm_pair_index])
        #     print(self.ecolog.vtilde_matrix_inv)

        #     print("===================================")
        #     import pdb

        #     pdb.set_trace()
        return arm1, arm2
