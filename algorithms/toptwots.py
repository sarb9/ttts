from algorithms.ecolog import ECOLOG
import numpy as np
from utils.utils import weighted_norm, gaussian_sample_ellipsoid_not_centered
from algorithms.utils import add_chosen_arm


class TopTwoTS(ECOLOG):
    def __init__(
        self, arms, true_param_norm_ub=None, failure_level=0.05, inflation=1, **kwargs
    ):
        super().__init__(arms, true_param_norm_ub, failure_level, **kwargs)
        self.inflation = inflation

    @add_chosen_arm
    def choose(self, context):
        # Record the theta
        self.thetas.append(self.ecolog.theta)
        # active_arms = self.armpairs[context]
        arms = self.arms[context]

        self.ecolog.update_ucb_bonus()
        param1 = gaussian_sample_ellipsoid_not_centered(
            self.ecolog.theta,
            self.ecolog.vtilde_matrix_inv,
            self.ecolog.conf_radius * self.inflation,
        )
        arm1 = np.argmax(np.dot(arms, param1))
        arm2 = arm1
        while arm2 == arm1:
            param2 = gaussian_sample_ellipsoid_not_centered(
                self.ecolog.theta,
                self.ecolog.vtilde_matrix_inv,
                self.ecolog.conf_radius * self.inflation,
            )

            arm2 = np.argmax(np.dot(arms, param2))
            if arm1 == 

        return arm1, arm2
