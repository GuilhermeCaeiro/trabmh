import numpy as np
import random
from individual import Individual
from utils import Utils
import time
import pdb 

class Initialization:

    def __init__(self):
        pass

    @classmethod
    def binary_random(self, environment, population_size):
        chromosomes = [] 

        for i in range(0, population_size):
            chromosome = np.zeros((environment.n, 1))
            chromosome[random.sample(range(0, environment.n), k = environment.s)] = 1
            chromosomes.append(chromosome)

        return chromosomes

    @classmethod
    def binary_biased(self, environment, population_size, min_diff):
        min_diff = float(min_diff)
        chromosomes = []

        while len(chromosomes) < population_size:    
            chromosome = np.zeros((environment.n, 1))
            chromosome[random.sample(range(0, environment.n), k = environment.s)] = 1
            
            found_close = False
            for existing_chromosome in chromosomes:
                if np.mean(np.abs(existing_chromosome - chromosome)) < min_diff:
                    found_close = True
                    break
            if not found_close:
                chromosomes.append(chromosome)

        return chromosomes
    
    @classmethod
    def permutation_random(self, environment, population_size):
        chromosomes = self.binary_random(environment, population_size)
        return Utils.convert_chromosomes_from_binary_to_permutation(chromosomes)

    @classmethod
    def permutation_biased(self, environment, population_size, min_diff):
        chromosomes = self.binary_biased(environment, population_size, min_diff)
        return Utils.convert_chromosomes_from_binary_to_permutation(chromosomes)

    @classmethod
    def Gabriel_heuristic_local_search(self, environment, population_size):
        # TODO Add the option to produce initial solutions using 
        # Gabriel's heuristics and local search.
        pass

    @classmethod
    def initialize_population(self, environment, population_size):
        function_and_params = environment.initialization_method.split("_")
        function_name = function_and_params[0]
        params = function_and_params[1:] if len(function_and_params) > 0 else []
        function_name = environment.encoding + "_" + function_name
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            population_chromosomes = func(environment, population_size, *params)
            population = []
            best_sol = - np.inf

            for chromosome in population_chromosomes:
                individual = Individual(
                    chromosome,
                    environment
                )
                if individual.fitness > best_sol:
                    best_sol = individual.fitness
                population.append(individual)

            return population, best_sol
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))