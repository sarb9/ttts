import numpy as np


class BanditFactory:
    def __init__(self, BANDIT, n_arms, n_contexts, d, **kwargs) -> None:
        self.BANDIT = BANDIT
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.d = d
        self.kwargs = kwargs

    def create_bandit(self, **kwargs):
        return self.BANDIT(
            self.n_arms, self.n_contexts, self.d, **self.kwargs, **kwargs
        )


class DuelBandit:
    def duel(self, arm1, arm2):
        pass

    def _initialize(self):
        pass

    def best_arm(self, context=None):
        pass

    def public_info(self):
        pass

    def next_context(self):
        return np.random.randint(0, self.n_contexts)

    def get_context(self):
        return self.context
