import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "optimisers", "ABC"))

from artificial_bee_colony import ArtificialBeeColony


def test_function(x):
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

plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("Artificial Bee Colony Convergence")
plt.grid(True)
plt.savefig("visualisations/abc_convergence.png")
plt.show()

print("Best position:", best_position)
print("Best fitness:", best_fitness)