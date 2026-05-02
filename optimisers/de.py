

from .base import  Optimiser
import numpy as np
import time
import random

class DE(Optimiser):
    def __init__(self, config, mutate_rate=0.8, crossp=0.7):
        super().__init__(config)
        self.mutate_rate = mutate_rate
        self.crossp = crossp

    def fitness(self, bot, params):
        return -bot.evaluate(params)  # minimise negative profit

    def optimise(self, bot):

        # bounds setup
        min_b, max_b = np.array(self.bounds).T
        diff = max_b - min_b

        # init population in [0,1]
        pop = np.random.rand(self.pop_size, self.dim)

        pop_denorm = min_b + pop * diff
        fitness = np.array([self.fitness(bot, p) for p in pop_denorm])

        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]

        for _ in self._iter_loop():

            for j in range(self.pop_size):

                idxs = [i for i in range(self.pop_size) if i != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                mutant = np.clip(a + self.mutate_rate * (b - c), 0, 1)

                cross = np.random.rand(self.dim) < self.crossp
                if not np.any(cross):
                    cross[np.random.randint(0, self.dim)] = True

                trial = np.where(cross, mutant, pop[j])

                trial = np.clip(trial, 0, 1)  # safety
                trial_denorm = min_b + trial * diff

                f = self.fitness(bot, trial_denorm)

                if f < fitness[j]:
                    pop[j] = trial
                    fitness[j] = f

                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm

            yield best, -fitness[best_idx]