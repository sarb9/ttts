class DuelInteraction:
    def __init__(self, bandit_factory, alg_factory) -> None:
        self.bandit_factory = bandit_factory
        self.alg_factory = alg_factory

        self.runs = []
        self.callbacks = {}

    def interact(
        self,
        n_steps,
        n_runs,
    ):
        for i in range(n_runs):
            env = self.bandit_factory.create_bandit()
            alg = self.alg_factory.create_alg()

            context = env.context
            for i in range(n_steps):
                arm1, arm2 = alg.get_arms(context)
                previous_context = context
                result, context = env.duel(arm1, arm2)
                alg.update(previous_context, arm1, arm2, result)
                if i % 100 == 0:
                    print(f"Step {i}: {ALG.__name__}")
        return alg

    def add_callback(self, name, callback):
        self.callbacks[name] = callback
