from .base import Optimiser
import numpy as np
import time


class PSO(Optimiser):

    def __init__(self, config, w=0.7, c1=1.5, c2=1.5, seed=42):
        super().__init__(config)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.default_rng(seed)

    def fitness(self, bot, params):
        return bot.evaluate(params)

    def optimise(self, bot):

        min_b, max_b = np.array(self.bounds).T

        # Initial population
        pop = self.rng.uniform(
            min_b,
            max_b,
            (self.pop_size, self.dim)
        )

        # Velocities
        velocities = np.zeros((self.pop_size, self.dim))

        # Personal bests
        p_best = pop.copy()

        p_fitness = np.array([
            self.fitness(bot, p)
            for p in pop
        ])

        # Global best
        best_idx = np.argmax(p_fitness)
        g_best = pop[best_idx].copy()
        g_fitness = p_fitness[best_idx]

        # Early stopping
        start_time = time.time()
        calls0 = bot.eval_count
        best_hist = [g_fitness]

        # Main loop
        for iteration in self._iter_loop():

            for i in range(self.pop_size):

                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)

                # Velocity update
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (p_best[i] - pop[i])
                    + self.c2 * r2 * (g_best - pop[i])
                )

                # Position update
                pop[i] += velocities[i]

                # Bound clipping
                pop[i] = np.clip(
                    pop[i],
                    min_b,
                    max_b
                )

                # Evaluate
                f = self.fitness(bot, pop[i])

                # Personal best update
                if f > p_fitness[i]:

                    p_best[i] = pop[i].copy()
                    p_fitness[i] = f

                    # Global best update
                    if f > g_fitness:

                        best_idx = i
                        g_best = pop[i].copy()
                        g_fitness = f

            best_hist.append(g_fitness)

            calls_made = bot.eval_count - calls0

            if self._should_stop(
                start_time,
                calls_made,
                best_hist
            ):
                break

            yield g_best, g_fitness