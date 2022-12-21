import random
from re import S
import time
import os
from selection import Selection
from individual import Individual
from initialization import Initialization
from search import Search
from plot import Plot
from utils import Utils
import math
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from datetime import datetime, date

class GeneticAlgoritm:
    def __init__(self, experiment):
        self.experiment = experiment
        self.start_time = None
        self.experiment_id = experiment["experiment_id"]
        self.execution_id = experiment["execution_id"]
        self.instance = experiment["instance"]
        self.seed = experiment["seed"]
        self.silent = experiment["silent"]
        self.generate_plots = experiment["generate_plots"]
        self.max_generations = experiment["max_generations"]
        self.max_generations_without_change = experiment["max_generations_without_change"]
        self.max_time = experiment["max_time"]
        self.max_time_without_change = experiment["max_time_without_change"]
        self.population_size = experiment["population_size"]
        self.n = experiment["n"]
        self.m = experiment["m"]
        self.s = experiment["s"]
        self.A = experiment["A"]
        self.R = experiment["R"]
        self.encoding = experiment["encoding_method"]
        self.initialization_method = experiment["initialization_method"]
        self.selection_method = experiment["selection_method"]
        self.mutation_method = experiment["mutation_method"]
        self.self_mutation = experiment["self_mutation"]
        self.crossover_method = experiment["crossover_method"]
        self.parent_selection_method = experiment["parent_selection_method"]
        self.mutation_probability = experiment["mutation_probability"]
        self.crossover_probability = experiment["crossover_probability"]
        self.perform_assexual_crossover = experiment["perform_assexual_crossover"]
        self.best_sol = -np.inf
        self.best_known_result = experiment["best_known_result"]
        self.elite_size = math.ceil(self.population_size * experiment["elite_size"])
        # offspring_size is a limit on how many children are going going to be generated.
        # Ultimately, it is the crossover_probability that will control if a mating attempt
        # will be successful or not. For that reason, offspring_size shouldn't be confused 
        # with a target number.
        # In other words, if crossover_probability = 1.0, offspring_size will be the number of
        # individuals to be generated. However, if crossover_probability < 1.0, the number of
        # individuals generated will probably be less than offspring_size.
        self.offspring_size = math.floor(self.population_size * experiment["offspring_size"])
        self.commoners_size = self.population_size - self.elite_size
        self.best_sol_tracking = []
        self.best_sol_changes = []
        self.best_sol_change_times = []
        self.best_sol_change_generations = []
        self.plots_dir = experiment["plots_dir"]
        self.perform_path_relinking = experiment["perform_path_relinking"]
        self.stop_message = "Maximum number of iterations reached"
        self.generations_ran = 0
        # Avoids repeated individuals in the population.
        # This option can be set by the user and, in the future, 
        # might be employed by adaptive methods to bring more
        # diversification to the population.
        self.avoid_clones = experiment["avoid_clones"] 
        self.time_until_adapt = experiment["time_until_adapt"]
        self.generations_until_adapt = experiment["generations_until_adapt"]
        self.perform_lagrangian = experiment["perform_lagrangian"]
        self.perform_adaptation = experiment["perform_adaptation"]
        self.total_time = None
        self.time_since_last_change = None
        self.generations_since_last_change = None
        self.local_search_method = experiment["local_search_method"]
        self.max_time_local_search = experiment["max_time_local_search"]
        self.perform_local_search = experiment["perform_local_search"]
        self.stop_on_target = experiment["stop_on_target"]
        self.stop_target = experiment["stop_target"]
        
        if self.seed == 0:
            self.seed = random.randint(0, 1000000)

        random.seed(self.seed)
        np.random.seed(self.seed)

    def loop(self):
        self.start_time = time.time()

        # generates initial population 
        population, self.best_sol = Initialization.initialize_population(self, self.population_size)
        self.best_sol_tracking.append(self.best_sol)
        self.best_sol_changes.append(self.best_sol)
        self.best_sol_change_times.append(time.time())
        self.best_sol_change_generations.append(1)
        elite = None
        pr_candidates = None
        self.log("Population initialized", population)

        for generation in range(self.max_generations):
            self.log("Generation", generation + 1)

            # complement population if necessary (mainly as a result of changes in the selection method)
            if len(population) != self.population_size:
                population, self.best_sol = self.complement_population(population, self.best_sol)

            # Sorts the population. It is required because some selection methods rely on it
            population = Utils.sort_population(population)

            # generates children 
            children = []
            children_generated = 0

            while children_generated < self.offspring_size:
                if random.uniform(0,1) < self.crossover_probability:
                    # Selects parents
                    parents = Selection.select(population, 2, self.parent_selection_method)
                    offspring = parents[0].breed(parents[1])
                    children = children + list(offspring)
                
                    if self.encoding == "binary":
                        # Usually, each successful attempt generates 2 children. For that reason,
                        # each unsuccessful attempt will also count as 2 attempts.
                        children_generated += 2
                    elif self.encoding == "permutation": 
                        # Permutations usually produce only one child.
                        children_generated += 1

                # Generates children in an asexual way, with probability self.mutation_probability 
                # of happening. The way it is executed right now might lead to offspring_size + 1 
                # individuals.
                if random.uniform(0,1) < self.mutation_probability:
                    an_individual = Selection.select(population, 1, self.parent_selection_method)
                    mutant = an_individual[0].mutate(self.self_mutation)
                    children = children + [mutant]
                    children_generated += 1

            self.log("Children produced", children)

            # Updates the population
            population = population + children # + mutants

            if self.avoid_clones:
                population = Utils.remove_repeated_individuals(population)

            self.log("Candidate population", population)

            # Sorts the population. It is required because some selection methods 
            # and some other things rely on it
            population = Utils.sort_population(population)

            # updates de best solution if necessary
            if population[0].fitness > self.best_sol:
                self.best_sol = population[0].fitness
                self.best_sol_changes.append(self.best_sol)
                self.best_sol_change_times.append(time.time())
                self.best_sol_change_generations.append(generation + 1)
            # Track the best solution found so far
            self.best_sol_tracking.append(self.best_sol)

            # selects individuals for the next generation
            elite, commoners = Utils.split_elite_commoners(population, self.elite_size)
            commoners = Selection.select(commoners, self.commoners_size, self.selection_method)
            population = elite + commoners
            self.log("Population after elitism and selection", population)
            
            # Calculates atrributes related to stopping criteria and adaptation
            self.compute_stop_and_adaptation_attributes(generation)

            if self.perform_adaptation: 
                self.adapt()

            self.generations_ran += 1

            # Checks stopping criteria and stops accordingly
            if self.stop_execution(generation):
                break

        # ===========================================================================================

        # Performs path relinking and local search
        if self.perform_path_relinking:
            pr_candidates = Utils.get_path_relinking_candidates(population, self.elite_size)
            new_individuals = self.path_relinking(pr_candidates)
            if len(new_individuals) > 0:
                population += new_individuals
                best_new_sol =  max([ind.fitness for ind in new_individuals])
                if best_new_sol > self.best_sol:
                    self.best_sol = best_new_sol
                    self.best_sol_changes.append(self.best_sol)
                    self.best_sol_change_times.append(time.time())
                    self.best_sol_change_generations.append(generation + 1)
                elite, commoners = Utils.split_elite_commoners(population, self.elite_size)
        
        if self.perform_local_search:
            ls_candidates = pr_candidates = Utils.get_path_relinking_candidates(population, self.elite_size)
            new_chromosomes, new_solutions = Search.local_search(self, ls_candidates, self.max_time_local_search, self.best_sol, self.local_search_method)
            
            if len(new_chromosomes) > 0:
                population += new_chromosomes
                if max(new_solutions) > self.best_sol:
                    self.best_sol = max(new_solutions)
                    self.best_sol_changes.append(self.best_sol)
                    self.best_sol_change_times.append(time.time())
                    self.best_sol_change_generations.append(generation + 1)
                elite, commoners = Utils.split_elite_commoners(population, self.elite_size)

        # Draw the plots.
        Plot.draw_plots(self)
        self.save_solution_times()

        print(f"Experiment {self.experiment_id} finished - Instance name: {self.instance} - Result {self.best_sol} - Best Known Result {self.best_known_result} - Gap {(self.best_sol - self.best_known_result)/self.best_known_result} - Time {self.total_time}.")

        sol_changes = {
            "best_sol_changes": self.best_sol_changes,
            "best_sol_change_times": self.best_sol_change_times,
            "best_sol_change_generations": self.best_sol_change_generations
        }

        # Updates the elite set.
        return elite, self.generations_ran, self.stop_message, self.seed, sol_changes, self.start_time

    # TODO add a stop criterion based on the optimality gap.

    def save_solution_times(self):
        if not self.generate_plots:
            return 

        file = os.path.join(self.plots_dir, "solution_times.csv")
        gaps = None

        if math.isinf(self.best_known_result):
            gaps = ["" for i in range(len(self.best_sol_changes))]
        else:
            gaps = [(-self.best_sol_changes[i] + self.best_known_result) / abs(self.best_known_result) for i in range(len(self.best_sol_changes))]

        df = pd.DataFrame({
            "best_sol_changes": self.best_sol_changes,
            "best_sol_change_times": self.best_sol_change_times,
            "best_sol_change_generations": self.best_sol_change_generations,
            "gaps": gaps
        })
        df["best_sol_change_times"] = df["best_sol_change_times"] - self.start_time
        df.to_csv(file, sep=";")
        

    def compute_stop_and_adaptation_attributes(self, current_generation):
        self.total_time = time.time() - self.start_time 
        self.time_since_last_change = time.time() - self.best_sol_change_times[-1]
        self.generations_since_last_change = current_generation + 1 - self.best_sol_change_generations[-1]

    def stop_execution(self, current_generation):
        if self.stop_on_target and self.best_sol > self.stop_target:
            self.stop_message ="Target solution reached."
            return True

        if current_generation == self.max_generations - 1:
            self.stop_message ="Maximum number of generations reached."

        if self.total_time >= self.max_time and self.max_time > 0:
            self.stop_message = "Maximum time reached"
            return True

        if self.time_since_last_change >= self.max_time_without_change and self.max_time_without_change > 0:
            self.stop_message = "Maximum time without improvement reached."
            return True

        if self.generations_since_last_change >= self.max_generations_without_change and self.max_generations_without_change > 0:
            self.stop_message = "Maximum number of generations without improvement reached"
            return True
        
            
        return False

    def adapt(self):
        if self.time_since_last_change > self.time_until_adapt or self.generations_since_last_change > self.generations_until_adapt:
                self.adapt_parameters()

        elif len(self.best_sol_tracking) > 3 and self.best_sol_tracking[-1] > self.best_sol_tracking[-2] and self.best_sol_tracking[-2] == self.best_sol_tracking[-3]:
            self.reset_parameters()

    def complement_population(self, population, best_sol):  
        if len(population) != self.population_size:
            if len(population) < self.population_size:
                new_individuals, highest_new_sol = Initialization.initialize_population(self, self.population_size - len(population))
                if highest_new_sol > best_sol:
                    best_sol = highest_new_sol
                population = population + new_individuals
        return population, best_sol

    def adapt_parameters(self,):
        # parameters to be adapted in that order:
        # n individuos elite - done
        # alteração do tamanho da população?
        # parâmetros dos métodos de seleção/crossover/mutação
        # alteração dos próprios métodos para métodos mais aleatórios/diversificadores
        # em ultimo caso métodos de reinicio de população
        if self.elite_size > math.ceil(self.population_size * self.experiment["elite_size"]/3):
            self.elite_size -= 1
        elif self.population_size < 2*self.experiment["population_size"]:
            if self.population_size == self.experiment["population_size"]:
                self.population_size += math.floor(self.experiment["population_size"] * 0.1)
                self.elite_size = math.ceil(self.population_size * self.experiment["elite_size"])
                self.offspring_size = math.ceil(self.population_size * self.experiment["offspring_size"])
                self.commoners_size = self.population_size - self.elite_size
            else:
                self.population_size += math.floor(self.experiment["population_size"] * 0.1)
        elif self.selection_method != "nbestdifferent":
            self.selection_method = "nbestdifferent"
        


    def reset_parameters(self,):
        self.population_size = self.experiment["population_size"]
        self.elite_size = math.ceil(self.population_size * self.experiment["elite_size"])
        self.offspring_size = math.floor(self.population_size * self.experiment["offspring_size"])
        self.commoners_size = self.population_size - self.elite_size
        self.selection_method = self.experiment["selection_method"]
        self.mutation_method = self.experiment["mutation_method"]
        self.self_mutation = self.experiment["self_mutation"]
        self.crossover_method = self.experiment["crossover_method"]
        self.parent_selection_method = self.experiment["parent_selection_method"]
        self.mutation_probability = self.experiment["mutation_probability"]
        self.crossover_probability = self.experiment["crossover_probability"]


    def path_relinking(self, population):

        if len(population) <= 1:
            return []

        intersting_sol_found = []
        uniques , indices = np.unique([individual.binary_chromosome for individual in population], return_index=True)
        unique_individuals = [population[index] for index in indices]

        for individual in unique_individuals:
            for other_individual in unique_individuals:
                diff = np.where(individual.binary_chromosome != other_individual.binary_chromosome)[0]
                if len(diff) > 0:
                    for i in range(0, len(diff)):
                        # forward path relinking
                        new_chromosome = individual.binary_chromosome.copy()
                        new_chromosome[diff[i]] = other_individual.binary_chromosome[diff[i]]
                        new_individual = Individual(new_chromosome, individual.environment)
                        if new_individual.fitness >= self.best_sol:
                            intersting_sol_found.append(new_individual)
        return intersting_sol_found
    
    def log(self, message, additional_content = "", status = "INFO"):
        if self.silent:
            return

        prepared_message = "{}" + message + "\033[0m"
        if status == "INFO":
            print(prepared_message.format("\033[96m"), additional_content)
        elif status == "ERROR":
            print(prepared_message.format("\033[91m"), additional_content)
        else: # warning
            print(prepared_message.format("\033[93m"), additional_content)