import numpy as np
from sklearn.linear_model import LogisticRegression

from algorithms.algorithms import ContextualAlgorithm
from algorithms.utils import add_chosen_arm


class Uniform(ContextualAlgorithm):
    def __init__(self, arms, **kwargs):
        self.arms = arms
        self.n_contexts = arms.shape[0]
        self.n_arms = arms.shape[1]
        self.d = arms.shape[2]

        self.features = []
        self.observations = []
        self.thetas = []
        self.chosen_arms = []

    def parameters(self):
        return {}

    @add_chosen_arm
    def choose(self, context):
        if len(self.thetas) == 0:
            self.thetas = [np.array([0, 0]) for i in range(10)]
        # Record estimates for plotting
        if len(self.features) >= 9:
            self.thetas.append(self.__get_theta_hat())

        # Selects two arms uniformly at random
        arms = np.random.choice(self.n_arms, 2, replace=False)
        return arms[0], arms[1]

    def learn(self, context, arm, observation):
        arm1, arm2 = arm
        arms = self.arms[context, arm1], self.arms[context, arm2]
        self.features.append(arms[0] - arms[1])
        self.observations.append(observation)

    def __get_theta_hat(self):
        # Returns the policy that finds the arm with the highest mean reward using logistic regression
        features = np.array(self.features)
        observations = np.array(self.observations)

        # Fit logistic regression
        lr = LogisticRegression(
            fit_intercept=False,
            penalty=None,
            solver="lbfgs",
            max_iter=10000,
            multi_class="ovr",
        )
        lr.fit(features, observations)
        return lr.coef_[0]

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
