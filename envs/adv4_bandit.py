import numpy as np

from envs.guassian_bandit import GuassianBandit


class Adv4Bandit(GuassianBandit):
    def __init__(self, n_arms, n_contexts, d, fit_into_unit_ball, **kwargs) -> None:
        # set the theta to (1, ..., 1)
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.d = d
        self.fit_into_unit_ball = fit_into_unit_ball

        self.theta = np.ones((self.d))

        self.arms = np.random.normal(size=(self.n_contexts, self.n_arms, self.d)) / (
            self.d**2
        )

        # set i'th arm's i'th feature to 1 in i'th context
        for i in range(self.d):
            self.arms[i % self.n_arms, i % self.n_contexts, i] = 4

        if self.fit_into_unit_ball:
            # cut the arms to fit into the unit ball
            for c in range(self.n_contexts):
                for a in range(self.n_arms):
                    if np.linalg.norm(self.arms[c, a]) > 1:
                        self.arms[c, a] /= np.linalg.norm(self.arms[c, a])

        self._parameters = {
            "n_arms": n_arms,
            "n_contexts": n_contexts,
            "d": d,
            "fit_into_unit_ball": fit_into_unit_ball,
            "arms": self.arms,
            "theta": self.theta,
        }

        self._next_context()
