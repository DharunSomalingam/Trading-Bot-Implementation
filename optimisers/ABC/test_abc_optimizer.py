from artificial_bee_colony import ArtificialBeeColony


def test_function(x):
    """
    Simple test function.
    Maximum value is 0 when all x values are close to 0.
    """
    return -sum(value ** 2 for value in x)


bounds = [(-10, 10), (-10, 10)]

abc = ArtificialBeeColony(
    objective_function=test_function,
    bounds=bounds,
    colony_size=30,
    max_iterations=100,
    limit=20,
    seed=42
)

best_position, best_fitness, history = abc.run()

print("Best position:", best_position)
print("Best fitness:", best_fitness)
print("Iterations:", len(history))