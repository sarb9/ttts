import numpy as np

from algorithms.algorithms import ContextualAlgorithm
from algorithms.utils import add_chosen_arm


def sherman_morrison(M, a):
    # Returns the inverse of the matrix M + a * a.T
    # Uses the Sherman-Morrison formula
    a = np.expand_dims(a, axis=1)
    M = M - (M @ a @ a.T @ M) / (1 + a.T @ M @ a)
    return M


class LinUniform(ContextualAlgorithm):
    def __init__(self, arms, reg=1, **kwargs):
        self.arms = arms
        self.n_contexts = arms.shape[0]
        self.n_arms = arms.shape[1]
        self.d = arms.shape[2]

        self.features = []
        self.observations = []
        self.thetas = [np.zeros(self.d)]
        self.chosen_arms = []

        self.reg = reg
        self.V_inv = (1 / self.reg) * np.eye(self.d)

    def parameters(self):
        return {}

    @add_chosen_arm
    def choose(self, context):
        arm = np.random.randint(self.n_arms)
        return arm

    def learn(self, context, arm, observation):
        feature = self.arms[context, arm]
        self.features.append(feature)
        self.observations.append(observation)

        # Update the inverse of the covariance matrix
        self.V_inv = sherman_morrison(self.V_inv, feature)
        self.thetas.append(self.__get_theta())

    def __get_theta(self):
        # Returns the current estimate of theta
        features = np.array(self.features)
        observations = np.array(self.observations)
        observations = np.expand_dims(observations, axis=1)
        return np.dot(self.V_inv, np.sum(features.T @ observations, axis=1))

    def get_policy(self, checkpoint=None):
        theta_hat = None
        if checkpoint is None:
            theta_hat = self.thetas[-1]
        else:
            theta_hat = self.thetas[checkpoint]

        def policy(context):
            # Returns the arm with the highest mean reward
            arms = self.arms[context]
            return np.argmax(np.dot(arms, theta_hat))

        return policy
