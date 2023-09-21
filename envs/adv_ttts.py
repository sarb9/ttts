import numpy as np

from envs.guassian_bandit import GuassianBandit


class AdvTTTS(GuassianBandit):
    def __init__(self, n_arms, n_contexts, d, **kwargs) -> None:
        # set the theta to (1, ..., 1)
        assert n_arms == 3, "This instance needs the number of arms to be 3"
        assert n_contexts == 1, "This instance needs the number of contexts to be 1"
        assert d == 2, "This instance nneds the number of dimensions to to be 2"
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.d = d

        self.theta = np.array([1, 0])

        self.arms = np.array(
            [
                [
                    [1, 0],
                    [0.99999, 0.004472124774634615],
                    [0, 1],
                ]
            ]
        )

        self._parameters = {
            "n_arms": n_arms,
            "n_contexts": n_contexts,
            "d": d,
            "arms": self.arms,
            "theta": self.theta,
        }

        self._next_context()
