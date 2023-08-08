import time


class CallBack:
    def initialize(self, experiment):
        self.reset()
        self.experiment = experiment

    def __call__(self, env_name, alg_name, env, alg, run):
        assert hasattr(self, "experiment")

    def reset(self):
        if hasattr(self, "experiment"):
            self.experiment = None

    def wrap_up(self):
        pass


class Experiment:
    def __init__(self, bandit_factories, algorithm_factories) -> None:
        self.bandit_factories = bandit_factories
        self.algorithm_factories = algorithm_factories

        self.callbacks = []

    def interact(
        self,
        n_steps,
        n_runs,
        log_progress=False,
    ):
        for callback in self.callbacks:
            callback.initialize(self)

        # log elappsed time
        if log_progress:
            start_time = time.time()
        for bandit_factory in self.bandit_factories:
            for run in range(n_runs):
                env = bandit_factory.create_bandit()
                for algorithm_factory in self.algorithm_factories:
                    alg = algorithm_factory.create_algorithm(**env.public_parameters())
                    env, alg = self.single_run(env, alg, n_steps)
                    for callback in self.callbacks:
                        callback(
                            bandit_factory.name,
                            algorithm_factory.name,
                            env,
                            alg,
                            run,
                        )
                if log_progress:
                    print(
                        f"Finished run {run + 1} of {n_runs} for {bandit_factory.name} in {time.time() - start_time} seconds"
                    )

        for callback in self.callbacks:
            callback.wrap_up()

    def single_run(self, env, alg, n_steps):
        for i in range(n_steps):
            context = env.get_context()
            arm = alg.choose(context)
            previous_context = context
            observation, context = env.pull(arm)
            alg.learn(previous_context, arm, observation)

        return env, alg

    def add_callback(self, callback):
        self.callbacks.append(callback)
