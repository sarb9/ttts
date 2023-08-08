from algorithms.ecolog import ECOLOG
import numpy as np
from utils.utils import weighted_norm, gaussian_sample_ellipsoid_not_centered


class EcoTS(ECOLOG):
    def choose(self, context):
        # Record the theta
        self.thetas.append(self.ecolog.theta)
        active_arms = self.armpairs[context]

        self.ecolog.update_ucb_bonus()
        param = gaussian_sample_ellipsoid_not_centered(
            self.ecolog.theta, self.ecolog.vtilde_matrix_inv, self.ecolog.conf_radius
        )

        # choose the arm with max estimated reward
        max_arm_pair_index = np.argmax(np.dot(active_arms, param))

        arm1 = max_arm_pair_index // (self.n_arms - 1)
        arm2 = max_arm_pair_index % (self.n_arms - 1)
        if arm2 >= arm1:
            arm2 += 1
        return arm1, arm2
