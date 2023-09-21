import numpy as np
import matplotlib.pyplot as plt

from experiment.experiment import CallBack


class ActionTracker(CallBack):
    def __init__(self) -> None:
        pass

    def initialize(self, experiment):
        super().initialize(experiment)
        self.experiment.action_tracker = {
            alg_fac.name: {
                bandit_fac.name: [] for bandit_fac in self.experiment.bandit_factories
            }
            for alg_fac in self.experiment.algorithm_factories
        }

    def __call__(self, env_name, alg_name, env, alg, run):
        # call the parent class's __call__ method
        super().__call__(env_name, alg_name, env, alg, run)
        chosen_arms = []
        for i in range(len(alg.chosen_arms)):
            arm1 = min(alg.chosen_arms[i])
            arm2 = max(alg.chosen_arms[i])
            arm_pair_code = str(arm1) + str(arm2)
            chosen_arms.append(arm_pair_code)
        self.experiment.action_tracker[alg_name][env_name].append(chosen_arms)

    def reset(self):
        if hasattr(self, "experiment"):
            self.experiment.action_tracker = None
        super().reset()

    def wrap_up(self):
        n_envs = len(self.experiment.bandit_factories)
        n_algs = len(self.experiment.algorithm_factories)
        fig = plt.figure(figsize=(16, 5 * n_algs))
        gs = fig.add_gridspec(n_envs * n_algs, hspace=0)
        axs = gs.subplots(sharex=True)
        fig.suptitle("Chosen Actions")

        for i, bandit_factory in enumerate(self.experiment.bandit_factories):
            n_arms = bandit_factory.factory_parameters["n_arms"]
            for j, alg_factory in enumerate(self.experiment.algorithm_factories):
                n_steps = len(
                    self.experiment.action_tracker[alg_factory.name][
                        bandit_factory.name
                    ][0]
                )
                chosen_arms = {
                    str(arm2) + str(arm1): np.zeros(n_steps)
                    for arm1 in range(n_arms)
                    for arm2 in range(arm1 + 1)
                }
                for ca in self.experiment.action_tracker[alg_factory.name][
                    bandit_factory.name
                ]:
                    for k, arm_pair in enumerate(ca):
                        chosen_arms[arm_pair][k] += 1

                ax = axs[i * (n_algs - 1) + j]
                ax.stackplot(
                    range(n_steps),
                    chosen_arms.values(),
                    labels=chosen_arms.keys(),
                    alpha=0.8,
                )
                ax.legend(loc="upper left", reverse=True)
                ax.set_title(f"{alg_factory.name} on {bandit_factory.name}")
                ax.set_xlabel("time step")
                ax.set_ylabel(f"{alg_factory.name} on {bandit_factory.name}")
                ax.legend()

        for ax in axs:
            ax.label_outer()
        plt.show()
