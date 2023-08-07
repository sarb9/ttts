import numpy as np


class BanditFactory:
    def __init__(self, bandit_class, name, **parameters) -> None:
        self.bandit_class = bandit_class
        self.name = name
        self.factory_parameters = parameters

    def create_bandit(self, **special_parameters):
        parameters = self.factory_parameters.copy()
        parameters.update(special_parameters)
        return self.bandit_class(**parameters)


class Bandit:
    def __init__(self, **parameters) -> None:
        pass

    def pull(self, arm):
        pass

    def public_parameters(self):
        pass

    def parameters(self):
        pass


class ContextualBandit(Bandit):
    def _next_context(self):
        self.context = np.random.randint(0, self.n_contexts)

    def get_context(self):
        ##### DON'T FORGET TO COPY THE CONTEXT!!!!!! #####
        return self.context
