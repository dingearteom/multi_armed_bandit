from abc import *
import numpy as np
from scipy.stats import beta


class Agent(object, metaclass=ABCMeta):
    def __init__(self, bandit):
        self.bandit = bandit

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def run_one_step(self):
        """Return the machine index to take action on AND obtained reward"""
        pass

    @abstractmethod
    def reset(self):
        pass


class EpsilonGreedyAgent(Agent):
    def __init__(self, bandit, eps):
        super(EpsilonGreedyAgent, self).__init__(bandit)

        assert 0. <= eps <= 1.0
        assert bandit is not None

        self.eps = eps
        self.reset()

    def __str__(self):
        return "epsilon greedy: {}".format(self.eps)

    def reset(self):
        # в качестве инициализации дернем за каждую ручку
        self.rewards = [self.bandit.generate_reward(i)
                        for i in range(self.bandit.n)]
        self.counts = [1] * self.bandit.n
        # self.rewards = [0] * self.bandit.n
        # self.counts = [0] * self.bandit.n

    def run_one_step(self):
        if np.random.random() < self.eps:
            # случайный выбор
            i = np.random.randint(0, self.bandit.n)
        else:
            # выбираем лучшего
            i = max(range(self.bandit.n),
                    key=lambda k: float(self.rewards[k]) / (self.counts[k]))

        r = self.bandit.generate_reward(i)
        self.rewards[i] += r
        self.counts[i] += 1
        # self.estimates[i] = float(self.rewards_sum[i]) / (self.counts[i] + 1)
        # self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])
        return i, r


class UCB1Agent(Agent):
    def __init__(self, bandit):
        super(UCB1Agent, self).__init__(bandit)
        self.reset()

    def __str__(self):
        return "UCB1"

    def reset(self):
        # в качестве инициализации дернем за каждую ручку
        self.rewards = [self.bandit.generate_reward(i)
                        for i in range(self.bandit.n)]
        self.counts = [1] * self.bandit.n
        # self.rewards = [0] * self.bandit.n
        # self.counts = [0] * self.bandit.n

    def get_estimates(self):
        mean_rewards = np.array(self.rewards, dtype=np.float) / self.counts
        additional_terms = np.sqrt(
            2. * np.log(np.sum(self.counts)) / self.counts)
        return mean_rewards + additional_terms

    def run_one_step(self):
        estimates = self.get_estimates()
        i = np.argmax(estimates)
        r = self.bandit.generate_reward(i)
        self.rewards[i] += r
        self.counts[i] += 1
        return i, r


class ThompsonSampling(Agent):
    def __int__(self, bandit):
        self.bandit = bandit
        self.reset()

    def reset(self):
        self.a = np.ones(self.bandit.n)
        self.b = np.ones(self.bandit.n)

    def __str__(self):
        return "ThompsonSampling"

    def run_one_step(self):
        samples = []
        for i in range(self.bandit.n):
            samples.append(np.random.beta(self.a[i], self.b[i]))

        i = np.argmax(samples)
        r = self.bandit.generate_reward(i)
        if r == 1:
            self.a[i] += 1
        else:
            self.b[i] += 1
        return i, r
