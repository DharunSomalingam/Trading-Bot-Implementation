import numpy as np


class ArtificialBeeColony:
    """
    Artificial Bee Colony optimiser for maximising trading bot fitness.
    """

    def __init__(self, objective_function, bounds, colony_size=30, max_iterations=100, limit=20, seed=42):
        self.objective_function = objective_function
        self.bounds = np.array(bounds, dtype=float)
        self.colony_size = colony_size
        self.food_sources = colony_size // 2
        self.max_iterations = max_iterations
        self.limit = limit
        self.random = np.random.default_rng(seed)

        self.dimensions = len(bounds)
        self.lower_bounds = self.bounds[:, 0]
        self.upper_bounds = self.bounds[:, 1]

        self.positions = self.random.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(self.food_sources, self.dimensions)
        )

        self.fitness = np.array([self.objective_function(position) for position in self.positions])
        self.trials = np.zeros(self.food_sources)

        best_index = np.argmax(self.fitness)
        self.best_position = self.positions[best_index].copy()
        self.best_fitness = self.fitness[best_index]
        self.history = [self.best_fitness]

    def generate_candidate(self, index):
        partner_index = self.random.integers(0, self.food_sources)

        while partner_index == index:
            partner_index = self.random.integers(0, self.food_sources)

        phi = self.random.uniform(-1, 1, self.dimensions)

        candidate = self.positions[index] + phi * (
            self.positions[index] - self.positions[partner_index]
        )

        return np.clip(candidate, self.lower_bounds, self.upper_bounds)

    def greedy_selection(self, index, candidate):
        candidate_fitness = self.objective_function(candidate)

        if candidate_fitness > self.fitness[index]:
            self.positions[index] = candidate
            self.fitness[index] = candidate_fitness
            self.trials[index] = 0
        else:
            self.trials[index] += 1

    def employed_bee_phase(self):
        for index in range(self.food_sources):
            candidate = self.generate_candidate(index)
            self.greedy_selection(index, candidate)

    def onlooker_bee_phase(self):
        adjusted_fitness = self.fitness - np.min(self.fitness) + 1e-9
        probabilities = adjusted_fitness / np.sum(adjusted_fitness)

        for _ in range(self.food_sources):
            selected_index = self.random.choice(self.food_sources, p=probabilities)
            candidate = self.generate_candidate(selected_index)
            self.greedy_selection(selected_index, candidate)

    def scout_bee_phase(self):
        for index in range(self.food_sources):
            if self.trials[index] >= self.limit:
                self.positions[index] = self.random.uniform(
                    self.lower_bounds,
                    self.upper_bounds
                )
                self.fitness[index] = self.objective_function(self.positions[index])
                self.trials[index] = 0

    def update_best_solution(self):
        best_index = np.argmax(self.fitness)

        if self.fitness[best_index] > self.best_fitness:
            self.best_fitness = self.fitness[best_index]
            self.best_position = self.positions[best_index].copy()

        self.history.append(self.best_fitness)

    def run(self):
        for _ in range(self.max_iterations):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()
            self.update_best_solution()

        return self.best_position, self.best_fitness, self.history