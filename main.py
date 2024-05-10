import numpy as np
import matplotlib.pyplot as plt
import copy
from creator import save_grid_to_file, plot_and_save_fitness, plot_and_save_heatmaps
from itertools import product
import pandas as pd

SUDOKU_FILE_PATH = "sudokus/grid_20_1.txt"

class Sudoku:
    def __init__(self, grid):
        self.grid = np.array(grid)

    def fitness(self):
        conflicts = 0
        for i in range(9):
            conflicts += self._count_conflicts(self.grid[i, :])
            conflicts += self._count_conflicts(self.grid[:, i])

        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                conflicts += self._count_conflicts(self.grid[i:i+3, j:j+3].flatten())

        return conflicts
      
    def _count_conflicts(self, grid):
      return len(grid) - len(set(grid))
 
      
class Individual:
    def __init__(self, chromosome):
        self.chromosome = np.array(chromosome)
        self.fitness = float('+inf')
        self.mutation_count = 0

    def mutate(self, base_grid, mutation_rate, mutation_count):
        rows, cols = self.chromosome.shape
        row_indices = np.random.permutation(rows)
        for i in row_indices:
            if self.mutation_count >= mutation_count:
                break  
            if np.random.rand() < mutation_rate:
                mutable_indices = [j for j in range(cols) if base_grid[i, j] == 0]
                if len(mutable_indices) >= 2: 
                    swap_indices = np.random.choice(mutable_indices, 2, replace=False)
                    
                    self.chromosome[i, swap_indices[0]], self.chromosome[i, swap_indices[1]] = \
                        self.chromosome[i, swap_indices[1]], self.chromosome[i, swap_indices[0]]
                        
                    self.mutation_count += 1  
                    
        self.mutation_count = 0
    
    def calculate_fitness(self):
        sudoku = Sudoku(self.chromosome)
        self.fitness = sudoku.fitness()
        return self.fitness

class Population:
    def __init__(self, size, initial_population, base_grid):
        self.individuals = initial_population
        self.size = size
        self.base_grid = base_grid

    def evaluate_fitness(self):
        for individual in self.individuals:
            individual.calculate_fitness()

    def selection(self, tournament_sizes):
        selected = []
        for _ in range(self.size):
            participants = np.random.choice(self.individuals, tournament_sizes)
            fittest = min(participants, key=lambda ind: ind.fitness)
            selected.append(fittest)
        self.individuals = selected

    def crossover(self, parent1, parent2):
        point = np.random.randint(1, 8)
        child1_chromosome = np.vstack((parent1.chromosome[:point], parent2.chromosome[point:]))
        child2_chromosome = np.vstack((parent2.chromosome[:point], parent1.chromosome[point:]))
        return Individual(child1_chromosome), Individual(child2_chromosome)

    def mutate(self, mutation_rate, mutation_count):
        for individual in self.individuals:
            individual.mutate(self.base_grid, mutation_rate, mutation_count)

class GeneticAlgorithm:
    def __init__(self, base_grid, population_size, generations, elitism_rate, tournament_size, mutation_rate, mutation_count):
        self.base_grid = base_grid
        self.generations = generations
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.mutation_count = mutation_count
        
        initial_population = self._generate_initial_population()
        self.population = Population(self.population_size, initial_population, self.base_grid)
        self.best_result_found = self.population.individuals[0]
        self.best_fitness_over_time = []
        self.solutions = []
        
    def _create_individual(self):
      individual = np.array(self.base_grid, dtype=int)
      for i in range(9):
          row = individual[i]
          if 0 in row:  # Check if there are empty cells to fill
              used_numbers = set(row[row != 0])
              missing_numbers = [num for num in range(1, 10) if num not in used_numbers]
              np.random.shuffle(missing_numbers)
              missing_index = 0
              for j in range(9):
                  if row[j] == 0:  # Fill only empty cells
                      row[j] = missing_numbers[missing_index]
                      missing_index += 1
      return Individual(individual)

    def _generate_initial_population(self):
        print("Generating initial population...")
        population = [self._create_individual() for _ in range(self.population_size)]
        print("Initial population generated")
        return population

    def run(self):
      for i in range(self.generations):
          self.population.evaluate_fitness()
          best_generation_result_found = min(self.population.individuals, key=lambda individual: individual.fitness)
          
          self.population.selection(self.tournament_size)
          
          if (self.best_result_found.fitness > best_generation_result_found.fitness):
            self.best_result_found = copy.copy(best_generation_result_found)
            print("Best result found is: ", self.best_result_found.fitness)
          
          # Keep a portion of the best individuals (elitism)
          elite_count = int(self.elitism_rate * len(self.population.individuals))  # For example, 15% elitism
          elites = sorted(self.population.individuals, key=lambda individual: individual.fitness)[:elite_count]
            
          if (elites[0].fitness == 2):
            save_grid_to_file(elites[0].chromosome, "best_2.txt")
            
          if (elites[0].fitness == 0):
            save_grid_to_file(elites[0].chromosome, "best_0.txt")
            self.solutions.append(elites[0].chromosome)
            break
          
          # print(f'Creating non-elite children for {i} generation...')
          new_individuals = elites  # Start the new generation with the elites
          while len(new_individuals) < self.population_size:
              parent1, parent2 = np.random.choice(self.population.individuals, 2, replace=False)
              child1, child2 = self.population.crossover(parent1, parent2)
              new_individuals.extend([child1, child2])
          self.population.individuals = new_individuals[:self.population_size]
          
          if (i != self.generations - 1):
            self.population.mutate(self.mutation_rate, self.mutation_count) 

          current_best_fitness = min(ind.fitness for ind in self.population.individuals)
          self.best_fitness_over_time.append(current_best_fitness)

    def best_solution(self):
        best = min(self.population.individuals, key=lambda individual: individual.fitness)
        return best
      
  
def main():
    tournament_sizes = [3, 5, 7, 9]
    mutation_rates = [0.15, 0.2, 0.25, 0.3]
    elitism_rates = [0.2, 0.3]
    mutation_counts = [1, 3, 5, 7, 9]
    population_size = 500
    generations = 100

    base_grid = np.loadtxt(SUDOKU_FILE_PATH, dtype=int)
    results = {}
    all_results = []

    for params in product(tournament_sizes, mutation_rates, elitism_rates, mutation_counts):
        t_size, m_rate, e_rate, m_count = params
        print(f"Running GA with TS={t_size}, MR={m_rate}, ER={e_rate}, MC={m_count}")
        ga = GeneticAlgorithm(base_grid, population_size, generations, e_rate, t_size, m_rate, m_count)
        ga.run()
        key = (t_size, m_rate, e_rate, m_count)
        results[key] = ga.best_fitness_over_time
        all_results.append([t_size, m_rate, e_rate, m_count, min(ga.best_fitness_over_time)])

    for idx, (params, fitness) in enumerate(results.items()):
        plot_and_save_fitness({params: fitness}, f"{params[0]}_{params[1]}_{params[2]}_{params[3]}")

    df = pd.DataFrame(all_results, columns=['t_size', 'm_rate', 'e_rate', 'm_count', 'min_fitness'])
    plot_and_save_heatmaps(df, 't_size', 'm_count', 'min_fitness', ['e_rate', 'm_rate'])

      
if __name__ == "__main__":
    main()

