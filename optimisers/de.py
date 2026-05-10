from .base import Optimiser
import numpy as np
import time

class DE(Optimiser):
    def __init__(self, config, mutate_rate=0.8, crossp=0.7):
        super().__init__(config)
        self.mutate_rate = mutate_rate
        self.crossp = crossp

    def fitness(self, bot, params):
        return bot.evaluate(params)

    def optimise(self, bot):

        min_b, max_b = np.array(self.bounds).T
        diff = max_b - min_b
        pop = np.random.rand(self.pop_size, self.dim)

        pop_denorm = min_b + pop * diff
        fitness = np.array([self.fitness(bot, p) for p in pop_denorm])

        best_idx = np.argmax(fitness)
        best = pop_denorm[best_idx]

        start_time = time.time()
        calls0 = bot.eval_count
        best_hist = [fitness[best_idx]]

        for iteration in self._iter_loop():

            for j in range(self.pop_size):

                idxs = [i for i in range(self.pop_size) if i != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                mutant = np.clip(a + self.mutate_rate * (b - c), 0, 1)

                cross = np.random.rand(self.dim) < self.crossp
                if not np.any(cross):
                    cross[np.random.randint(0, self.dim)] = True

                trial = np.where(cross, mutant, pop[j])

                trial = np.clip(trial, 0, 1)
                trial_denorm = min_b + trial * diff

                f = self.fitness(bot, trial_denorm)

                if f > fitness[j]:
                    pop[j] = trial
                    fitness[j] = f

                    if f > fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm

            best_hist.append(fitness[best_idx])
            calls_made = bot.eval_count - calls0

            if self._should_stop(start_time, calls_made, best_hist):
                break

            yield best, fitness[best_idx]