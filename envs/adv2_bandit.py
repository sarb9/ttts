import numpy as np
from envs.guass_bandit import DuelBandit


class Adv2Bandit(DuelBandit):
    def __init__(self, n_arms, n_contexts, d, adv=False, **kwargs) -> None:
        super().__init__(n_arms, n_contexts, d, **kwargs)
        self.adv = adv

    def reset(self, fit_into_unit_ball=False):
        # set arms features to zero in the beginning
        self.arms = np.random.normal(size=(self.n_contexts, self.n_arms, self.d))
        self.arms[:, :, 0] = 0
        self.arms[0, 0, 0] = 1
        if fit_into_unit_ball:
            # cut the arms to fit into the unit ball
            for c in range(self.n_contexts):
                for a in range(self.n_arms):
                    if np.linalg.norm(self.arms[c, a]) > 1:
                        self.arms[c, a] /= np.linalg.norm(self.arms[c, a])

        # set the hidden variable to (1, 0, ..., 0)
        self.theta = np.zeros((self.d))
        self.theta[0] = 5

        # set the current context
        self.context = np.random.randint(0, self.n_contexts)

        return self.context
