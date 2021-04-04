import pickle
import numpy as np
from copy import deepcopy
import math


class Drawer:

    @staticmethod
    def draw(named_bandit, agents, type, ax, steps, alpha=0.5):
        assert type == 'avg' or type == 'cum'

        for agent in agents:
            if type == 'avg':
                file = f'data/bandit_{named_bandit.name}/avg_regret/{str(agent(named_bandit.bandit))}.pickle'
            else:
                file = f'data/bandit_{named_bandit.name}/regret/{str(agent(named_bandit.bandit))}.pickle'

            with open(file, 'rb') as fp:
                regret = pickle.load(fp)

            regret = np.array(regret)
            regret = regret[:, :steps]
            num_experiments, num_steps = regret.shape

            lower_bound = []
            upper_bound = []
            for i in range(num_steps):
                a = deepcopy(regret[:, i])
                a = sorted(a)
                lower_bound.append(a[int((num_experiments * (1 - alpha)) / 2)])

                r = math.ceil((num_experiments * (1 + alpha)) / 2)
                r = min(r, num_experiments - 1)
                upper_bound.append(a[r])

            mean_avg_regret = np.mean(regret, axis=0)
            ax.plot(range(num_steps), mean_avg_regret, label=f'{str(agent(named_bandit.bandit))}')
            ax.fill_between(range(num_steps), lower_bound, upper_bound, alpha=0.2)