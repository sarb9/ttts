import numpy as np

from envs.bandits import ContextualBandit

THETA = None


class CloseArms2D(ContextualBandit):
    def __init__(
        self,
        n_arms,
        n_contexts,
        d,
        fit_into_unit_ball,
        **kwargs,
    ) -> None:
        assert n_arms == 3, "This instance needs the number of arms to be 3"
        assert n_contexts == 1, "This instance needs the number of contexts to be 1"
        assert d == 2, "This instance nneds the number of dimensions to to be 2"

        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.d = d
        self.fit_into_unit_ball = fit_into_unit_ball

        self.theta = np.array([1, 0])

        self.arms = np.array(
            [
                [
                    [10, 0],
                    [9.9999, 0.04472124774634615],
                    [0, 10],
                ]
            ]
        )

        self._parameters = {
            "n_arms": n_arms,
            "n_contexts": n_contexts,
            "d": d,
            "fit_into_unit_ball": fit_into_unit_ball,
            "arms": self.arms,
            "theta": self.theta,
        }

        # set the current context
        self._next_context()

    def pull(self, arm):
        # retun 1 with probability p if arm1 > arm2, 0 otherwise
        feature = self.arms[self.context, arm]
        reward = np.dot(feature, self.theta)

        # add sub-guassian noise
        noise = np.random.normal()

        self._next_context()
        return reward + noise, self.context

    def best_arm(self, context=None):
        if context is None:
            context = self.context
        return np.argmax(np.dot(self.arms[context], self.theta))

    def public_parameters(self):
        public_parameters = {
            "arms": self.arms.copy(),
            "true_param_norm_ub": np.linalg.norm(self.theta) + 1,
        }
        return public_parameters

    def parameters(self):
        return self._parameters
