from .base import Optimiser
import numpy as np
import time


class ABC(Optimiser):
    """
    Artificial Bee Colony optimiser for trading strategy parameter tuning.

    Each food source represents one candidate parameter set.
    Fitness is calculated by the TradingBot using bot.evaluate(params).
    """

    def __init__(self, config, limit=10):
        super().__init__(config)
        self.limit = limit

    def fitness(self, bot, params):
        return bot.evaluate(params)

    def _normalise_to_bounds(self, position, min_b, max_b):
        return np.clip(position, min_b, max_b)

    def optimise(self, bot):
        min_b, max_b = np.array(self.bounds).T
        diff = max_b - min_b

        # Initial food sources
        foods = min_b + np.random.rand(self.pop_size, self.dim) * diff
        fitness = np.array([self.fitness(bot, food) for food in foods])
        trials = np.zeros(self.pop_size)

        best_idx = np.argmax(fitness)
        best = foods[best_idx].copy()
        best_fitness = fitness[best_idx]

        start_time = time.time()
        calls0 = bot.eval_count
        best_hist = [best_fitness]

        for iteration in self._iter_loop():

            # Employed bee phase
            for i in range(self.pop_size):
                k = np.random.choice([idx for idx in range(self.pop_size) if idx != i])
                phi = np.random.uniform(-1, 1, self.dim)

                candidate = foods[i] + phi * (foods[i] - foods[k])
                candidate = self._normalise_to_bounds(candidate, min_b, max_b)

                candidate_fitness = self.fitness(bot, candidate)

                if candidate_fitness > fitness[i]:
                    foods[i] = candidate
                    fitness[i] = candidate_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Onlooker bee phase
            shifted_fitness = fitness - np.min(fitness) + 1e-9
            probabilities = shifted_fitness / np.sum(shifted_fitness)

            for _ in range(self.pop_size):
                i = np.random.choice(self.pop_size, p=probabilities)
                k = np.random.choice([idx for idx in range(self.pop_size) if idx != i])
                phi = np.random.uniform(-1, 1, self.dim)

                candidate = foods[i] + phi * (foods[i] - foods[k])
                candidate = self._normalise_to_bounds(candidate, min_b, max_b)

                candidate_fitness = self.fitness(bot, candidate)

                if candidate_fitness > fitness[i]:
                    foods[i] = candidate
                    fitness[i] = candidate_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Scout bee phase
            for i in range(self.pop_size):
                if trials[i] >= self.limit:
                    foods[i] = min_b + np.random.rand(self.dim) * diff
                    fitness[i] = self.fitness(bot, foods[i])
                    trials[i] = 0

            current_best_idx = np.argmax(fitness)

            if fitness[current_best_idx] > best_fitness:
                best_idx = current_best_idx
                best = foods[best_idx].copy()
                best_fitness = fitness[best_idx]

            best_hist.append(best_fitness)

            calls_made = bot.eval_count - calls0
            if self._should_stop(start_time, calls_made, best_hist):
                break

            yield best, best_fitness