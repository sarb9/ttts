import numpy as np
from envs.duel_bandit import DuelBandit


class AdvBandit3(DuelBandit):
    def __init__(
        self,
        n_arms,
        n_contexts,
        d,
        **kwargs,
    ) -> None:
        super().__init__(n_arms, n_contexts, d, **kwargs)
        print(n_contexts, d)
        assert n_contexts < d, "n_contexts must be less than d for AdvBandit3"

    def reset(self, fit_into_unit_ball=False):
        # set the theta to (0, 0, ..., 0, 1, ..., 1)
        self.theta = np.ones((self.d))
        self.theta[self.n_contexts :] = 1

        self.arms = np.random.normal(size=(self.n_contexts, self.n_arms, self.d))

        # set arms features to zero after the theta_non_zero_dimension
        self.arms[:, :, self.n_contexts :] = 0

        # set i'th arm's i'th feature to 1 in i'th context
        for i in range(self.n_contexts):
            self.arms[i, i, i] = 10

        if fit_into_unit_ball:
            # cut the arms to fit into the unit ball
            for c in range(self.n_contexts):
                for a in range(self.n_arms):
                    if np.linalg.norm(self.arms[c, a]) > 1:
                        self.arms[c, a] /= np.linalg.norm(self.arms[c, a])

        self.context = self.next_context()
        return self.context
