class AlgorithmFactory:
    def __init__(self, ALG, obversing_info, **kwargs) -> None:
        self.ALG = ALG
        self.obversing_info = obversing_info
        self.kwargs = kwargs

    def create_alg(self, **kwargs):
        return self.ALG(self.obversing_info, **self.kwargs, **kwargs)


class DuelAlgorithm:
    def get_arms(self, context):
        pass

    def learn(self, context, arm1, arm2, observation):
        pass

    def subscribed_values(self):
        pass

    def get_policy(self, sufficient_statistics=None):
        pass
