import numpy as np

from envs.bandits import ContextualBandit

THETA = None


class LinGuassianBandit(ContextualBandit):
    def __init__(
        self,
        n_arms,
        n_contexts,
        d,
        fit_into_unit_ball,
        fixed_theta=False,
        **kwargs,
    ) -> None:
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.d = d
        self.fit_into_unit_ball = fit_into_unit_ball

        # set arms features randomly
        self.arms = np.random.normal(size=(self.n_contexts, self.n_arms, self.d))
        if self.fit_into_unit_ball:
            self.arms /= np.linalg.norm(self.arms, axis=2, keepdims=True)

        # set the hidden variable randomly
        if fixed_theta:
            global THETA
            if THETA is None:
                THETA = np.random.normal(size=(self.d), scale=5 / np.sqrt(self.d))
            self.theta = THETA
        else:
            self.theta = np.random.normal(size=(self.d))

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
