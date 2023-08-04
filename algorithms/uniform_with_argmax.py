from algorithms.uniform import Uniform
import numpy as np


class UniformWithArgmax(Uniform):
    def get_arms(self, context):
        if len(self.features) < 10:
            # Call parent method if no data has been collected yet
            return super().get_arms(context)

        theta = self.get_theta_hat()
        self.thetas.append(theta)

        # Selects the arm with the highest estimated reward for the given context
        arms = self.arms[context]
        arm1 = np.argmax(arms @ theta)

        # Selects an arm uniformly at random from the remaining arms
        arm2 = np.random.choice([i for i in range(self.n_arms) if i != arm1])
        return arm1, arm2
