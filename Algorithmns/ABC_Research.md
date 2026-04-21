Artificial Bee Colony (ABC) Research

Artificial Bee Colony (ABC) is a swarm intelligence optimisation algorithm inspired by the foraging behaviour of honey bees. In the algorithm, candidate solutions are represented as food sources, and the quality of each solution is represented by the nectar amount. The search process is carried out by three types of bees: employed bees, onlooker bees, and scout bees. Employed bees exploit known food sources and generate neighbouring solutions, onlooker bees probabilistically choose promising food sources based on shared fitness information, and scout bees randomly search for new areas when an existing source is abandoned. This structure gives ABC a strong balance between exploitation of good solutions and exploration of new regions in the search space.

A key strength of ABC is its ability to avoid premature convergence and escape local optima. The neighbourhood update rule modifies a solution by comparing it against another randomly selected solution, and the step size naturally shrinks as solutions become closer, which helps the search move from broad exploration toward refined local improvement. In addition, the scout bee mechanism introduces diversity by replacing abandoned food sources with randomly generated new ones, preventing the population from stagnating around poor local minima.

The original motivation for ABC was to provide a simple but powerful alternative to other population-based optimisation techniques for numerical optimisation. The literature shows that the method was later extended to constrained optimisation and engineering design problems by incorporating simple constraint-handling rules, especially Deb’s feasibility rules. This allowed ABC to be applied not only to unconstrained benchmark functions but also to real engineering optimisation tasks where feasible and infeasible regions must be managed carefully.

ABC is especially relevant to the trading bot project because trading strategy optimisation is a high-dimensional, noisy, and non-convex problem. In this project, the trading bot uses technical analysis indicators such as SMA and EMA crossover signals, and the optimisation task is to search for the best strategy parameters, such as moving average window sizes, using historical Bitcoin OHLCV data rather than manually guessing them. Since the assignment frames trading bot construction as a multi-dimensional optimisation problem, ABC is a suitable candidate because it is population-based, adaptive, and designed for difficult global search spaces.

Compared with algorithms such as Genetic Algorithm (GA), Particle Swarm Optimisation (PSO), and Differential Evolution (DE), ABC is attractive because of its conceptual simplicity and its explicit exploration mechanism through scouts. GA relies more on crossover and mutation, PSO on velocity-position updates and social learning, and DE on vector differences between solutions, while ABC combines probabilistic selection with local perturbation and random re-initialisation. Prior studies reported that ABC performed competitively, and in some numerical optimisation settings outperformed GA, PSO, and PS-EA. This suggests strong promise for parameter tuning tasks such as trading strategy optimisation, although actual performance for the project will still need to be verified through implementation and backtesting.

Overall, ABC appears to be a strong algorithm for the project because it offers a good balance between simplicity, global search capability, and robustness against local minima. Its bee-role structure maps neatly to the exploration–exploitation trade-off required in trading strategy optimisation, making it a promising method to test in Phase 2 of the trading bot implementation. 



# Artificial Bee Colony (ABC) Algorithm – Synopsis

## 1. Introduction & Motivation

Artificial Bee Colony (ABC) is a **swarm intelligence optimisation algorithm** inspired by the foraging behaviour of honey bees. It is designed to solve **complex, high-dimensional optimisation problems**, where traditional deterministic methods struggle due to non-linearity, discontinuity, or multiple local optima .

In the context of this project, trading bot optimisation is a **multi-dimensional optimisation problem**, where parameters such as indicator windows and thresholds must be tuned for best performance . ABC is particularly suitable because it can efficiently search large solution spaces without requiring gradient information.


## 2. Key Idea of ABC Algorithm

ABC models the **collective intelligence of a bee colony**, where different types of bees cooperate to find the best food sources (solutions).

### Bee Roles

* **Employed Bees**
  Explore known food sources (solutions) and locally improve them
* **Onlooker Bees**
  Select promising solutions based on fitness (nectar quality)
* **Scout Bees**
  Randomly search for new solutions to maintain diversity

Each **food source = candidate solution**, and
**nectar amount = fitness value** .



## 3. Exploration vs Exploitation Balance

ABC maintains a strong balance between:

* **Exploitation (local search)**
  → Employed & onlooker bees refine existing solutions
* **Exploration (global search)**
  → Scout bees introduce randomness

This balance is crucial because:

* Too much exploitation → stuck in local minima
* Too much exploration → slow convergence

ABC dynamically adjusts step sizes during search, meaning:

* Large steps early → exploration
* Smaller steps later → fine-tuning 



## 4. Mechanism to Escape Local Optima

One of ABC’s strongest features is its **ability to escape local minima**:

* If a solution does not improve for a number of cycles (**“limit” parameter**), it is **abandoned**
* The corresponding employed bee becomes a **scout**
* A **new random solution is generated**

This prevents premature convergence and ensures continuous search of new regions .


## 5. Convergence Behaviour

ABC shows:

* **Fast initial convergence** (due to global exploration)
* **Stable convergence later** (due to greedy selection and local refinement)
* Adaptive step size reduces as solutions improve

However:

* Convergence may be **slower than DE or PSO** in some cases
* Performance depends on parameters like:

  * Population size (SN)
  * Limit
  * Maximum cycles (MCN)

---

## 6. Strengths and Weaknesses

### Strengths

* Simple and easy to implement
* Good balance of exploration & exploitation
* Strong ability to avoid local minima
* Works well for **non-linear and high-dimensional problems**
* Does not require gradient information

### Weaknesses

* Sensitive to parameter tuning (limit, population size)
* May converge slower than DE in some problems
* Random search (scout phase) can reduce efficiency if not controlled

---

## 7. Comparison with Other Algorithms

| Algorithm                       | Key Idea                        | Strength                            | Weakness                    |
| ------------------------------- | ------------------------------- | ----------------------------------- | --------------------------- |
| **GA (Genetic Algorithm)**      | Evolution (crossover, mutation) | Strong global search                | Can lose diversity          |
| **PSO (Particle Swarm)**        | Social learning                 | Fast convergence                    | Gets stuck in local minima  |
| **DE (Differential Evolution)** | Vector differences              | Strong optimisation power           | Parameter sensitive         |
| **ABC**                         | Bee foraging behaviour          | Good exploration + escape mechanism | Slightly slower convergence |

Studies show ABC performs competitively and sometimes **outperforms GA, PSO, and DE** on numerical optimisation problems .

---

## 8. Suitability for Trading Strategy Optimisation

ABC is highly suitable for this project because:

* Trading involves **optimising multiple parameters simultaneously**
* Market data is:

  * Noisy
  * Non-linear
  * Full of local optima

ABC can:

* Efficiently search parameter combinations
* Avoid overfitting to local patterns
* Adapt to complex search spaces

This aligns directly with the project goal of **optimising trading strategies using AI techniques** .

---

## 9. Conclusion (Why Choose ABC)

Artificial Bee Colony is a **powerful, flexible, and robust optimisation algorithm** inspired by natural swarm behaviour. Its ability to:

* Balance exploration and exploitation
* Escape local minima
* Handle complex, high-dimensional problems

makes it an excellent choice for **AI-based trading bot optimisation**.

Compared to other algorithms, ABC provides a **good trade-off between solution quality and robustness**, making it particularly effective for real-world optimisation problems like financial trading.


