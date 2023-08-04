import numpy as np


class DuelBandit:
    def __init__(self, n_arms, n_contexts, d) -> None:
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.d = d

        self.reset()

    def duel(self, arm1, arm2):
        # retun 1 with probability p if arm1 > arm2, 0 otherwise
        actual_feature = self.arms[self.context, arm1] - self.arms[self.context, arm2]
        pp = np.exp(np.dot(actual_feature, self.theta))
        p = pp / (1 + pp)
        obs = np.random.choice([0, 1], p=[1 - p, p])

        self.context = np.random.randint(0, self.n_contexts)
        return obs, self.context

    def best_arm(self, context=None):
        if context is None:
            context = self.context
        return np.argmax(np.dot(self.arms[context], self.theta))

    def true_param_norm_up(self):
        return np.linalg.norm(self.theta) + 1

    def reset(self, fit_into_unit_ball=False):
        # set arms features randomly
        self.arms = np.random.normal(size=(self.n_contexts, self.n_arms, self.d))
        if fit_into_unit_ball:
            self.arms /= np.linalg.norm(self.arms, axis=2, keepdims=True)

        # set the hidden variable randomly
        self.theta = np.random.normal(size=(self.d))

        # set the current context
        self.context = np.random.randint(0, self.n_contexts)

        return self.context
