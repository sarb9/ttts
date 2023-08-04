import numpy as np

from envs.guass_bandit import GuassianBandit


class AdvBandit4(GuassianBandit):
    def _initialize(self):
        # set the theta to (1, ..., 1)
        self.theta = np.ones((self.d))

        self.arms = np.random.normal(size=(self.n_contexts, self.n_arms, self.d)) / (
            self.d**2
        )

        # set i'th arm's i'th feature to 1 in i'th context
        for i in range(self.n_contexts):
            self.arms[i, i % self.n_contexts, i % self.d] = 4

        if self.fit_into_unit_ball:
            # cut the arms to fit into the unit ball
            for c in range(self.n_contexts):
                for a in range(self.n_arms):
                    if np.linalg.norm(self.arms[c, a]) > 1:
                        self.arms[c, a] /= np.linalg.norm(self.arms[c, a])

        self.context = self.next_context()
        return self.context
