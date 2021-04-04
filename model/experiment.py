import numpy as np


class Experiment(object):
    def __init__(self, bandit, agent):
        self.bandit = bandit
        self.agent = agent(bandit)
        self.reset()

    def __str__(self):
        return 'Experiment: "{}" vs "{}"'.format(str(self.agent), str(self.bandit))

    def reset(self):
        self.actions = []  # A list of machine ids

        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.
        self.avg_regrets = [0.]

        self.regret_smooth = 0.
        self.regrets_smooth = [0.]  # History of cumulative regret.
        self.avg_regrets_smooth = [0.]

        self.regret_alternative = 0.
        self.regrets_alternative = [0.]
        self.avg_regrets_alternative = [0.]

        self.agent.reset()

    def update_regret(self, i, r):
        # i (int): index of the selected machine.
        # r (float): obtained reward
        self.regret += self.bandit.generate_optimal_reward() - r
        self.regret_alternative += self.bandit.best_proba - r
        self.regret_smooth += self.bandit.best_proba - self.bandit.probas[i]
        self.regrets.append(self.regret)
        self.regrets_alternative.append(self.regret_alternative)
        self.regrets_smooth.append(self.regret_smooth)

        n = len(self.regrets)
        self.avg_regrets.append(self.regret / n)
        self.avg_regrets_alternative.append(self.regret_alternative / n)
        self.avg_regrets_smooth.append(self.regret_smooth / n)

    def run(self, num_steps):
        assert self.bandit is not None
        assert self.agent is not None
        self.reset()
        for _ in range(num_steps):
            i, r = self.agent.run_one_step()
            self.actions.append(i)
            self.update_regret(i, r)