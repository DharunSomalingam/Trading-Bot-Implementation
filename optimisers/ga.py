from .base import Optimiser
import numpy as np
import time


class GA(Optimiser):
    """
    Genetic Algorithm optimiser.
    
    Algorithm:
      1. Initialise random population within bounds
      2. Each generation:
           a. Elitism  — carry top N individuals unchanged
           b. Selection — tournament selection picks parents
           c. Crossover — uniform crossover produces child
           d. Mutation  — Gaussian noise applied per gene
      3. Restart half the population if no improvement for
         `restart_patience` generations (avoids premature convergence)
    """

    def __init__(
        self,
        config,
        crossover_rate=0.8,
        mutation_rate=0.4,
        mutation_scale=0.4,
        elitism=2,
        tournament_k=3,
        restart_patience=20,
    ):
        super().__init__(config)
        self.crossover_rate   = crossover_rate
        self.mutation_rate    = mutation_rate
        self.mutation_scale   = mutation_scale  # std dev as fraction of param range
        self.elitism          = elitism         # top-N carried over unchanged
        self.tournament_k     = tournament_k    # individuals compared per selection
        self.restart_patience = restart_patience
        self.rng              = np.random.default_rng(self.seed)

    # ── Fitness ────────────────────────────────────────────────────────

    def fitness(self, bot, params):
        return bot.evaluate(params)

    # ── Selection ──────────────────────────────────────────────────────

    def _tournament_select(self, population, fitness):
        """Pick tournament_k random individuals, return the fittest."""
        indices = self.rng.choice(len(population), self.tournament_k, replace=False)
        best    = indices[np.argmax(fitness[indices])]
        return population[best].copy()

    # ── Crossover ──────────────────────────────────────────────────────

    def _uniform_crossover(self, parent_a, parent_b):
        """Each gene independently drawn from either parent (50/50)."""
        mask = self.rng.random(self.dim) < 0.5
        return np.where(mask, parent_a, parent_b)

    # ── Mutation ───────────────────────────────────────────────────────

    def _gaussian_mutate(self, individual, min_b, max_b):
        """Add Gaussian noise to each gene with probability mutation_rate."""
        child  = individual.copy()
        ranges = max_b - min_b
        for i in range(self.dim):
            if self.rng.random() < self.mutation_rate:
                noise    = self.rng.normal(0, self.mutation_scale * ranges[i])
                child[i] = np.clip(child[i] + noise, min_b[i], max_b[i])
        return child

    # ── One generation ─────────────────────────────────────────────────

    def _evolve(self, population, fitness, min_b, max_b):
        """Produce next generation via elitism → crossover → mutation."""
        order      = np.argsort(fitness)[::-1]
        sorted_pop = population[order]
        new_pop    = []

        # Elitism: best individuals survive unchanged
        for i in range(self.elitism):
            new_pop.append(sorted_pop[i].copy())

        # Fill rest with crossover + mutation
        while len(new_pop) < self.pop_size:
            parent_a = self._tournament_select(population, fitness)
            parent_b = self._tournament_select(population, fitness)

            if self.rng.random() < self.crossover_rate:
                child = self._uniform_crossover(parent_a, parent_b)
            else:
                child = parent_a.copy()

            child = self._gaussian_mutate(child, min_b, max_b)
            new_pop.append(child)

        return np.array(new_pop)

    # ── Main optimisation loop ─────────────────────────────────────────

    def optimise(self, bot):
        """
        Generator — yields (best_params, best_fitness) after each generation.
        """
        min_b, max_b = np.array(self.bounds).T

        # Initialise random population within bounds
        population = self.rng.uniform(min_b, max_b, size=(self.pop_size, self.dim))
        fitness    = np.array([self.fitness(bot, p) for p in population])

        best_idx  = np.argmax(fitness)
        best      = population[best_idx].copy()
        best_fit  = fitness[best_idx]
        best_hist = [best_fit]

        no_improve = 0
        start_time = time.time()
        calls0     = bot.eval_count

        for _ in self._iter_loop():

            # ── Evolve ──────────────────────────────────────────────
            population = self._evolve(population, fitness, min_b, max_b)
            fitness    = np.array([self.fitness(bot, p) for p in population])

            # ── Track best ──────────────────────────────────────────
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fit:
                best_fit   = fitness[gen_best_idx]
                best       = population[gen_best_idx].copy()
                no_improve = 0
            else:
                no_improve += 1

            # ── Restart half population if stuck ────────────────────
            if no_improve >= self.restart_patience:
                half = self.pop_size // 2
                population[half:] = self.rng.uniform(
                    min_b, max_b, size=(self.pop_size - half, self.dim)
                )
                fitness[half:] = np.array([
                    self.fitness(bot, p) for p in population[half:]
                ])
                no_improve = 0

            best_hist.append(best_fit)

            # ── Check stopping conditions (from base class) ──────────
            calls_made = bot.eval_count - calls0
            if self._should_stop(start_time, calls_made, best_hist):
                break

            yield best, best_fit