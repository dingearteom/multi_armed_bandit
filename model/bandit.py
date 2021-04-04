from abc import *
import numpy as np


class Bandit(object, metaclass=ABCMeta):
    @abstractmethod
    def generate_reward(self, i):
        raise NotImplementedError

    @abstractmethod
    def generate_optimal_reward(self):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class BernoulliBandit(Bandit):

    def __init__(self, n=None, probas=None):
        assert probas is None or (n is None) or (len(probas) == n)
        if n is not None:
            self.n = n
        else:
            self.n = len(probas)
        if probas is None:
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        self.best_proba_i = np.argmax(self.probas)
        self.best_proba = self.probas[self.best_proba_i]

    def __str__(self):
        return "BernoulliBandit: [" + ", ".join(["{:.2f}".format(p) for p in self.probas]) + "]"

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probas[i]:
            return 1
        else:
            return 0

    def generate_optimal_reward(self):
        return self.generate_reward(self.best_proba_i)


class NamedBandit:
    def __init__(self, bandit, name):
        self.bandit = bandit
        self.name = name