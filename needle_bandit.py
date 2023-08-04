import numpy as np
from duel_bandit import DuelBandit


class NeedleBandit(DuelBandit):
    def reset(self, fit_into_unit_ball=False):
        # set arms features to zero in the beginning
        self.arms = np.random.normal(size=(self.n_contexts, self.n_arms, self.d)) / (
            self.d**4
        )
        self.arms[:, 0] = np.ones((self.n_contexts, self.d))
        if fit_into_unit_ball:
            self.arms /= np.linalg.norm(self.arms, axis=2, keepdims=True)

        # set the hidden variable randomly
        self.theta = np.ones((self.d)) / np.sqrt(self.d)

        # set the current context
        self.context = np.random.randint(0, self.n_contexts)

        return self.context


needle = NeedleBandit(3, 3, 3)
