import random

import numpy as np
from scipy.optimize import minimize

class GeneticAlgorithm:
    def __init__(
        self,
        fitness_function,
        parameter,
        lower_bound,
        upper_bound,
        population_size=40,
        mutation_rate=0.1,
        elitism_rate=0.1,
        crossover_probability=0.7,
        superhuman = True,
    ):
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.parameter_length = len(parameter)
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.crossover_probability = crossover_probability
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population = []
        self.superhuman = superhuman

    def initialize_population(self):
        self.population = [
            self.generate_random_parameter() for _ in range(self.population_size)
        ]

    def generate_random_parameter(self):
        random.seed(40)
        return [
            random.uniform(lower, upper)
            for lower, upper in zip(self.lower_bound, self.upper_bound)
        ]

    def supermutation(self, elite_individuals, steps=20):
        refined_elites = []
        
        for elite in elite_individuals:
            result = minimize(
                self.fitness_function,
                elite,
                method='Nelder-Mead',
                options={'maxiter':steps}
            )
            refined_elites.append(result.x)  # Store the optimized solution
        
        return refined_elites

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_probability:
                d = abs(np.array(parent1) - np.array(parent2))
                alpha = 0.5
                lower = np.minimum(parent1, parent2) - alpha * d
                upper = np.maximum(parent1, parent2) + alpha * d
                child1 = random.uniform(lower, upper)
                child2 = random.uniform(lower, upper)
                return list(child1), list(child2)
        else:
            return parent1, parent2

    def mutate(self, parameter):
        mutated_parameter = []
        for param, lower, upper in zip(parameter, self.lower_bound, self.upper_bound):
            if random.random() < self.mutation_rate:
                param += random.uniform(-0.1, 0.1)
                param = max(lower, min(upper, param))
            mutated_parameter.append(param)
        return mutated_parameter


    def select_elite(self, fitness_scores):
        sorted_indices = sorted(
            range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=False
        )
        elite_count = int(self.elitism_rate * len(fitness_scores))
        return [self.population[i] for i in sorted_indices[:elite_count]]

    def evolve_with_elites(self, fitness_function, generations, elite_callback=None):
        best_solution = None
        best_fitness = float("inf")
        for gen in range(generations):
            fitness_scores = [
                fitness_function(parameter) for parameter in self.population
            ]
            elite = self.select_elite(fitness_scores)
            if self.superhuman:
                elite = self.supermutation(elite)
            best_index = np.argmin(fitness_scores)
            if fitness_scores[best_index] < best_fitness:
                best_fitness = fitness_scores[best_index]
                best_solution = self.population[best_index]
            offspring = elite[:]
            while len(offspring) < self.population_size:
                parent1, parent2 = random.choices(elite, k=2)
                child1, child2 = self.crossover(parent1, parent2)
                offspring.extend([self.mutate(child1), self.mutate(child2)])
            self.population = offspring
            if elite_callback:
                elite_callback(gen, fitness_function(elite[0]))
        return best_solution

    def evolve_with_common(self, fitness_function, generations, elite_callback=None):
        best_solution = None
        best_fitness = float("inf")
        for gen in range(generations):
            fitness_scores = [
                fitness_function(parameter) for parameter in self.population
            ]
            if self.superhuman:
                elite = self.supermutation(elite)
            elite = self.select_elite(fitness_scores)
            best_index = np.argmin(fitness_scores)
            if fitness_scores[best_index] < best_fitness:
                best_fitness = fitness_scores[best_index]
                best_solution = self.population[best_index]
            offspring = elite[:]
            while len(offspring) < self.population_size:
                parent1, parent2 = random.choices(self.population, k=2)
                child1, child2 = self.crossover(parent1, parent2)
                offspring.extend([self.mutate(child1), self.mutate(child2)])
            self.population = offspring
            if elite_callback:
                elite_callback(gen, fitness_function(elite[0]))
        return best_solution

    def evolve_with_mix(self, fitness_function, generations, elite_callback=None):
        best_solution = None
        best_fitness = float("inf")
        for gen in range(generations):
            fitness_scores = [
                fitness_function(parameter) for parameter in self.population
            ]
            elite = self.select_elite(fitness_scores)
            best_index = np.argmin(fitness_scores)
            if fitness_scores[best_index] < best_fitness:
                best_fitness = fitness_scores[best_index]
                best_solution = self.population[best_index]
            if self.superhuman:
                elite = self.supermutation(elite)
            offspring = elite[:]
            while len(offspring) < self.population_size:
                parent1 = random.choice(
                    elite
                )  # Changed to random.choice as random.choices returns a list
                parent2 = random.choice(
                    self.population
                )  # Changed to random.choice as random.choices returns a list

                if random.random() < self.crossover_probability:
                    child1, child2 = self.crossover(parent1, parent2)
                    offspring.extend([self.mutate(child1), self.mutate(child2)])

            self.population = elite + offspring
            if elite_callback:
                elite_callback(gen, fitness_function(elite[0]))
        return best_solution

    def evolve_with_steady(self, fitness_function, generations, elite_callback=None):
        best_solution = None
        best_fitness = float("inf")

        for gen in range(generations):
            fitness_scores = [
                fitness_function(parameter) for parameter in self.population
            ]
            elite = self.select_elite(fitness_scores)
            if self.superhuman:
                elite = self.supermutation(elite)
            # Find the best solution in the current generation
            best_index = np.argmin(fitness_scores)
            if fitness_scores[best_index] < best_fitness:
                best_fitness = fitness_scores[best_index]
                best_solution = self.population[best_index]

            # Generate offspring
            offspring = []
            while len(offspring) < (len(self.population) - len(elite)):
                parent1, parent2 = random.choices(self.population, k=2)
                child1, child2 = self.crossover(parent1, parent2)
                offspring.extend([self.mutate(child1), self.mutate(child2)])

            # Create a combined population of non-elite and offspring
            combined_population = [
                ind for ind in self.population if ind not in elite
            ] + offspring

            # Select a portion of combined population to replace the current population
            new_population = elite[:]  # Save elite

            num_to_select = self.population_size - len(elite)
            new_population.extend(random.sample(combined_population, num_to_select))

            self.population = new_population

            # Call the elite callback if provided
            if elite_callback:
                elite_callback(gen, fitness_function(elite[0]))

        return best_solution


def elite_callback(gen, elite):
    print(f"Generation {gen}: Top Elite - {elite}")


def optimization_GA(mytarget, initial_guess, bounds):
    lower_bound = [item[0] for item in bounds]
    upper_bound = [item[1] for item in bounds]
    ga = GeneticAlgorithm(
        mytarget,
        initial_guess,
        lower_bound,
        upper_bound,
        population_size=30,
        mutation_rate=0.1,
        elitism_rate=0.1,
        crossover_probability=0.8,
        superhuman=True,
    )
    ga.initialize_population()
    best_solution = ga.evolve_with_mix(
        mytarget, generations=30, elite_callback=elite_callback
    )

    return best_solution


# def f(x):
#    #return np.sum(np.array(X)**2)
#    n = len(x)
#    sum1 = np.sum(np.square(x))
#    sum2 = np.sum(np.cos(2 * np.pi * np.array(x)))
#    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e
#


# Example usage
# population_size = 100
# initial_guess = [0] * 2
# mutation_rate = 0.1
# elitism_rate = 0.1
# crossover_probability = 0.9
# generations = 100
# lower_bound = [0] * len(initial_guess)
# upper_bound = [10] * len(initial_guess)
#
# ga = GeneticAlgorithm(f,initial_guess, population_size=population_size, mutation_rate=mutation_rate, elitism_rate=elitism_rate, crossover_probability=crossover_probability, lower_bound=lower_bound, upper_bound=upper_bound)
# ga.initialize_population()
# best_solution=ga.evolve_with_elites(f, generations, elite_callback)
# best_solution=ga.evolve_with_common(f, generations, elite_callback)
# best_solution=ga.evolve_with_mix(f, generations, elite_callback)
# best_solution=ga.evolve_with_steady(f, generations, elite_callback)
# print("Best solution found:", best_solution)
