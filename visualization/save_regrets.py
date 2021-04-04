from model.experiment import Experiment
import pickle
import os
from tqdm import tqdm
from shutil import rmtree


class SaveRegrets:
    def __init__(self, experiments):
        self.named_bandits, self.agents = zip(*experiments)
        self.reset()

    def reset(self):
        for named_bandit in self.named_bandits:
            try:
                rmtree(f'data/bandit_{named_bandit.name}')
            except Exception as exc:
                pass

    def run(self, num_experiments=100, num_steps=50000):

        progress_bar = tqdm(total=len(self.agents) * num_experiments)
        for named_bandit, agent in zip(self.named_bandits, self.agents):
            regret = []
            avg_regret = []
            for j in range(num_experiments):
                exp = Experiment(named_bandit.bandit, agent)
                exp.run(num_steps)
                regret.append(exp.regrets_alternative)
                avg_regret.append(exp.avg_regrets_alternative)
                progress_bar.update(1)

            dir = f'data/bandit_{named_bandit.name}/regret'
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(f'{dir}/{str(agent(named_bandit.bandit))}.pickle', 'wb') as fp:
                pickle.dump(regret, fp)

            dir = f'data/bandit_{named_bandit.name}/avg_regret'
            if not os.path.exists(dir):
                os.makedirs(dir)

            with open(f'{dir}/{str(agent(named_bandit.bandit))}.pickle', 'wb') as fp:
                pickle.dump(avg_regret, fp)




