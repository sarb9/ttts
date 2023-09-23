import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from experiment.experiment import CallBack

CMPAS = [
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
]

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


class ThetaTracker(CallBack):
    def __init__(self, interval, style="seperate", show_actions=False) -> None:
        self.interval = interval
        assert style in ["seperate", "together"], "style must be seperate or together"
        self.style = style
        self.show_actions = show_actions

    def initialize(self, experiment):
        super().initialize(experiment)
        self.experiment.theta_tracker = {
            alg_fac.name: {
                bandit_fac.name: [] for bandit_fac in self.experiment.bandit_factories
            }
            for alg_fac in self.experiment.algorithm_factories
        }
        self.experiment.theta_tracker_true_theta = {
            alg_fac.name: {
                bandit_fac.name: [] for bandit_fac in self.experiment.bandit_factories
            }
            for alg_fac in self.experiment.algorithm_factories
        }
        self.experiment.theta_tracker_arms = {}

    def __call__(self, env_name, alg_name, env, alg, run):
        # call the parent class's __call__ method
        super().__call__(env_name, alg_name, env, alg, run)
        thetas = alg.thetas
        self.experiment.theta_tracker[alg_name][env_name].append(thetas)
        self.experiment.theta_tracker_true_theta[alg_name][env_name].append(env.theta)
        self.experiment.theta_tracker_arms[env_name] = env.arms

    def reset(self):
        if hasattr(self, "experiment"):
            self.experiment.theta_tracker = None
        super().reset()

    def wrap_up(self):
        if self.style == "seperate":
            self.seperate_wrap_up()
        elif self.style == "together":
            self.together_wrap_up()

    def together_wrap_up(self):
        n_envs = len(self.experiment.bandit_factories)
        n_algs = len(self.experiment.algorithm_factories)
        fig, axs = plt.subplots(n_envs)
        fig.suptitle("Thetas")
        fig.set_size_inches(9, 9 * n_envs)

        for i, bandit_factory in enumerate(self.experiment.bandit_factories):
            if n_envs == 1:
                ax = axs
            else:
                ax = axs[i]

            alg_name = self.experiment.algorithm_factories[0].name

            # Make sure all true thetas are the same
            assert np.all(
                self.experiment.theta_tracker_true_theta[alg_name][bandit_factory.name]
                == self.experiment.theta_tracker_true_theta[alg_name][
                    bandit_factory.name
                ][0]
            ), "For theta tracker callback all the true thetas should be the same"

            true_theta = self.experiment.theta_tracker_true_theta[alg_name][
                bandit_factory.name
            ][0]

            ax.set(xlabel="x", ylabel="y")
            ax.set_title(f"{bandit_factory.name}")
            legends = []

            # set xlim and ylim
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)

            if self.show_actions:
                all_arms = self.experiment.theta_tracker_arms[bandit_factory.name]
                for context in range(all_arms.shape[0]):
                    color = COLORS[context]
                    arms = all_arms[context]
                    origin = np.array([[0] * arms.shape[0], [0] * arms.shape[0]])

                    ax.quiver(
                        *origin,
                        arms[:, 0],
                        arms[:, 1],
                        color=color,
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                    )

            for j, alg_factory in enumerate(self.experiment.algorithm_factories):
                thetas = self.experiment.theta_tracker[alg_factory.name][
                    bandit_factory.name
                ]
                thetas = np.array(thetas)
                thetas = thetas[:, :: self.interval]
                thetas = np.mean(thetas, axis=0)

                num_points = thetas.shape[0]
                colors = np.arange(num_points)
                cmap = plt.get_cmap(CMPAS[j])

                for k in range(num_points):
                    ax.scatter(
                        thetas[k][0],
                        thetas[k][1],
                        c=cmap(colors[k] / num_points),
                        s=25,
                        alpha=0.85,  # * (k / num_points),
                    )
                legend = mpatches.Patch(
                    color=cmap(100),
                    alpha=1,
                    label=f"{alg_factory.name}",
                )
                legends.append(legend)

            ax.scatter(true_theta[0], true_theta[1], marker="*")
            ax.legend(handles=legends, loc="upper left", reverse=True)

        plt.show()

    def seperate_wrap_up(self):
        n_envs = len(self.experiment.bandit_factories)
        n_algs = len(self.experiment.algorithm_factories)
        fig, axs = plt.subplots((n_algs * n_envs + 1) // 2, 2)
        fig.suptitle("Thetas")
        fig.set_size_inches(16, 7 * ((n_algs * n_envs + 1) // 2))

        for i, bandit_factory in enumerate(self.experiment.bandit_factories):
            for j, alg_factory in enumerate(self.experiment.algorithm_factories):
                thetas = self.experiment.theta_tracker[alg_factory.name][
                    bandit_factory.name
                ]
                thetas = np.array(thetas)
                thetas = thetas[:, :: self.interval]
                thetas = np.mean(thetas, axis=0)

                ax = axs[((i * n_algs) + j) // 2, ((i * n_algs) + j) % 2]
                num_points = thetas.shape[0]
                colors = np.arange(num_points)
                cmap = plt.get_cmap("YlGn")

                for k in range(num_points):
                    ax.scatter(
                        thetas[k][0],
                        thetas[k][1],
                        c=cmap(colors[k] / num_points),
                        s=50,
                        alpha=0.7 * (k / num_points),
                    )
                # Make sure all true thetas are the same
                assert np.all(
                    self.experiment.theta_tracker_true_theta[alg_factory.name][
                        bandit_factory.name
                    ]
                    == self.experiment.theta_tracker_true_theta[alg_factory.name][
                        bandit_factory.name
                    ][0]
                ), "For theta tracker callback all the true thetas should be the same"

                true_theta = self.experiment.theta_tracker_true_theta[alg_factory.name][
                    bandit_factory.name
                ][0]
                ax.scatter(true_theta[0], true_theta[1], marker="*")

                ax.set(xlabel="x", ylabel="y")
                ax.set_title(f"{alg_factory.name} on {bandit_factory.name}")
                ax.legend(loc="upper left", reverse=True)

            ax.scatter(true_theta[0], true_theta[1], marker="*")

        # for ax in axs.flat:
        # ax.set(xlabel="x", ylabel="y")
        # for ax in axs.flat:
        # ax.label_outer()

        plt.show()
