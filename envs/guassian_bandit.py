import numpy as np

from envs.bandits import ContextualBandit


class GuassianBandit(ContextualBandit):
    def __init__(self, n_arms, n_contexts, d, fit_into_unit_ball, **kwargs) -> None:
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.d = d
        self.fit_into_unit_ball = fit_into_unit_ball

        # set arms features randomly
        self.arms = np.random.normal(size=(self.n_contexts, self.n_arms, self.d))
        if self.fit_into_unit_ball:
            self.arms /= np.linalg.norm(self.arms, axis=2, keepdims=True)

        # set the hidden variable randomly
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
        arm1, arm2 = arm
        # retun 1 with probability p if arm1 > arm2, 0 otherwise
        actual_feature = self.arms[self.context, arm1] - self.arms[self.context, arm2]
        pp = np.exp(np.dot(actual_feature, self.theta))
        p = pp / (1 + pp)
        obs = np.random.choice([0, 1], p=[1 - p, p])

        self._next_context()
        return obs, self.context

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
