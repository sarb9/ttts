import numpy as np

from envs.bandit_factory import DuelBandit


class GuassianBandit(DuelBandit):
    def __init__(self, n_arms, n_contexts, d, fit_into_unit_ball, **kwargs) -> None:
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.d = d
        self.fit_into_unit_ball = fit_into_unit_ball

        self._initialize()

    def _initialize(self):
        # set arms features randomly
        self.arms = np.random.normal(size=(self.n_contexts, self.n_arms, self.d))
        if self.fit_into_unit_ball:
            self.arms /= np.linalg.norm(self.arms, axis=2, keepdims=True)

        # set the hidden variable randomly
        self.theta = np.random.normal(size=(self.d))

        # set the current context
        self.context = self.next_context()

    def duel(self, arm1, arm2):
        # retun 1 with probability p if arm1 > arm2, 0 otherwise
        actual_feature = self.arms[self.context, arm1] - self.arms[self.context, arm2]
        pp = np.exp(np.dot(actual_feature, self.theta))
        p = pp / (1 + pp)
        obs = np.random.choice([0, 1], p=[1 - p, p])

        self.context = self.next_context()
        return obs, self.context

    def best_arm(self, context=None):
        if context is None:
            context = self.context
        return np.argmax(np.dot(self.arms[context], self.theta))

    def public_info(self):
        info = {"true_param_norm_up": np.linalg.norm(self.theta) + 1}
        return info
