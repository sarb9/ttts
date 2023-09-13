import numpy as np
import matplotlib.pyplot as plt

from experiment.experiment import CallBack


class SimpleRegret(CallBack):
    def __init__(self, interval) -> None:
        self.interval = interval

    def initialize(self, experiment):
        super().initialize(experiment)
        self.experiment.simple_regret = {
            alg_fac.name: {
                bandit_fac.name: [] for bandit_fac in self.experiment.bandit_factories
            }
            for alg_fac in self.experiment.algorithm_factories
        }

    def __call__(self, env_name, alg_name, env, alg, run):
        # call the parent class's __call__ method
        super().__call__(env_name, alg_name, env, alg, run)
        regrets = []
        for i in range(len(alg.thetas)):
            if i % self.interval == 0:
                policy = alg.get_policy(checkpoint=i)
                regret = 0
                for context in range(env.n_contexts):
                    best_arm = env.best_arm(context)
                    chosen_arm = policy(context)
                    regret += np.dot(
                        (env.arms[context, best_arm] - env.arms[context, chosen_arm]),
                        env.theta,
                    )
                assert regret >= 0
                regrets.append(regret / env.n_contexts)
                # so at the end, self.experiment.simple_regret
                # is a dictionary of lists of lists
                # where the outer list is indexed by by runs,
                # and the inner list is indexed by checkpoints.
        self.experiment.simple_regret[alg_name][env_name].append(regrets)

    def reset(self):
        if hasattr(self, "experiment"):
            self.experiment.simple_regret = None
        super().reset()

    def wrap_up(self):
        n_envs = len(self.experiment.bandit_factories)
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(n_envs, hspace=0)
        axs = gs.subplots(sharex=True)
        fig.suptitle("Simple Regrets")
        for i, bandit_factory in enumerate(self.experiment.bandit_factories):
            for alg_factory in self.experiment.algorithm_factories:
                regrets = np.array(
                    self.experiment.simple_regret[alg_factory.name][bandit_factory.name]
                )  # shape: (n_runs, n_checkpoints)
                means = np.mean(regrets, axis=0)
                stds = np.std(regrets, axis=0)
                ax = axs[i] if n_envs > 1 else axs
                ax.plot(
                    [i * self.interval for i in range(len(means))],
                    means,
                    label=f"{alg_factory.name} on {bandit_factory.name}",
                )
                ax.fill_between(
                    [i * self.interval for i in range(len(means))],
                    means - stds,
                    means + stds,
                    alpha=0.2,
                )
                ax.legend()
        plt.xlabel("Checkpoints")
        plt.ylabel("Simple Regrets")
        # Hide x labels and tick labels for all but bottom plot.
        if n_envs > 1:
            for ax in axs:
                ax.label_outer()
        plt.show()
