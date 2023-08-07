class AlgorithmFactory:
    def __init__(self, algorithm_class, name, needed_parameters, **parameters) -> None:
        self.algorithm_class = algorithm_class
        self.name = name
        self.needed_parameters = needed_parameters
        self.factory_paramters = parameters

    def create_algorithm(self, **env_parameters):
        parameters = self.factory_paramters.copy()
        parameters.update(env_parameters)
        return self.algorithm_class(**parameters)


class Algorithm:
    def __init__(self, env_parameters, **parameters) -> None:
        pass

    def choose(self):
        pass

    def learn(self, arm, observation):
        pass


class ContextualAlgorithm(Algorithm):
    def choose(self, context):
        pass

    def learn(self, context, arm, observation):
        pass
