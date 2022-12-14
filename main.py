import mat73
import numpy as np
import pandas as pd
import sys
import os
from scipy.io import loadmat
import json
import time
from datetime import datetime
from genetic_algorithm import GeneticAlgoritm
from utils import Utils
import pdb
from multiprocessing import Process, Pool, freeze_support, Lock, cpu_count
from itertools import repeat
from copy import deepcopy
import itertools
import cpuinfo

mutex = Lock()

# Checks if the param combination is valid.
# It focuses only on params that can be incompatible 
# between themselves.
def validade_experiment_params(params):
    # TODO implement param validation

    # if an initialization method is restricted to one type of encoding, the output can be converted to the other
    initialization_methods = ["random", "biased", "biasedweighted", "heuristics"] 
    selection_methods = ["roulette", "tournament", "ranking", "byclass", "fullyrandom", "nbest", "nbestdifferent"]
    binary_crossover_methods = ["singlepoint", "mask"]
    binary_mutation_methods = ["singlepointlagrangian", "singlepoint", "percentchange", "variablepercentchange"]
    permutation_crossover_methods = ["opx","lox"]
    permutation_mutation_methods = ["singleexchange", "percentchange", "variablepercentchange"]

    encoding_method = params["encoding_method"]
    crossover_methods = None
    mutation_methods = None
    
    if encoding_method == "binary": 
        crossover_methods = binary_crossover_methods
        mutation_methods = binary_mutation_methods
    elif encoding_method == "permutation": 
        crossover_methods = permutation_crossover_methods
        mutation_methods = permutation_mutation_methods
    else:
        return False, "Unknown encoding method {}.".format(encoding_method)

    initialization_method = params["initialization_method"].split("_")[0]
    if initialization_method not in initialization_methods:
        return False, "Unknown initialization method {} or it is incompatible with the encoding method {}.".format(initialization_method, encoding_method)

    crossover_method = params["crossover_method"].split("_")[0]
    if crossover_method not in crossover_methods:
        return False, "Unknown crossover method {} or it is incompatible with the encoding method {}.".format(crossover_method, encoding_method)

    mutation_method = params["mutation_method"].split("_")[0]
    if mutation_method not in mutation_methods:
        return False, "Unknown mutation method {} or it is incompatible with the encoding method {}.".format(mutation_method, encoding_method)
    
    # checks if the genetic operators exist
    selection_method = params["selection_method"].split("_")[0]
    if selection_method not in selection_methods:
        return False, "Unknown selection method {}.".format(selection_method)

    if params["s"] > params["n"]:
        return False, "s={} is greater than n={}. It must to be s <= n.".format(params["s"], params["n"])

    return True, ""

def build_experiments(experiment_setup):
    experiments = []
    experiment_count = 0

    instance_paths = os.listdir(experiment_setup["instance_dir"])

    for instance_path in sorted(instance_paths):
        instance_name = os.path.basename(instance_path)
        execution_id = str(int(time.time()))

        # If len(experiment_setup["instances"]) == 0, all instances are considered.
        # If different, only those in experiment_setup["instances"] will be considered.
        # Obs.: instance names are the full file name, including the extension. In other
        # words, "Instance_40_1.mat" is correct, while "Instance_40_1" is wrong.
        if len(experiment_setup["instances"]) > 0 and instance_name not in experiment_setup["instances"]:
            # TODO insert a log call here.
            continue

        instance = loadmat(os.path.join(experiment_setup["instance_dir"], instance_path))
        
        A = instance["A"]
        n = A.shape[0]
        m = A.shape[1]

        s_values = [int(value) for value in experiment_setup["s_values"]]
        if len(s_values) == 0:
            print("No specific 's' value specified through the parameter 's_values'. Assuming n/2.")
            s_values.append(int(n/2))

        best_known_result = - np.inf
        plots_dir = os.path.join(experiment_setup["plots_dir"], str(execution_id))

        try:
            # creates folder to store plots.
            if experiment_setup["generate_plots"]:
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)
        except Exception as e:
            print (e)


        try:
            tmp = instance_name.replace("Instance_", "")
            known_results_file = os.path.join(experiment_setup["known_results_dir"], "x_ls_" + tmp)
            known_result = mat73.loadmat(known_results_file)
            best_known_result = np.linalg.slogdet(np.matmul(np.matmul(A.T, np.diagflat(known_result['x_ls'])), A))[1]

        except Exception as e:
            print(e)

        for (encoding, max_generations, population_size, 
            initialization, selection, mutation, 
            self_mutation, crossover, parent_selection, 
            mutation_probability, crossover_probability, assexual_crossover, 
            elite_size, offspring_size, path_relinking, 
            time_until_adapt, generations_until_adapt, perform_lagrangian, 
            perform_adaptation, local_search_method, max_time_local_search,
            perform_local_search, s) in itertools.product(
                experiment_setup["encoding_method"], experiment_setup["max_generations"], experiment_setup["population_size"], 
                experiment_setup["initialization_method"], experiment_setup["selection_method"], experiment_setup["mutation_method"], 
                experiment_setup["self_mutation"], experiment_setup["crossover_method"], experiment_setup["parent_selection_method"], 
                experiment_setup["mutation_probability"], experiment_setup["crossover_probability"], experiment_setup["perform_assexual_crossover"], 
                experiment_setup["elite_size"], experiment_setup["offspring_size"], experiment_setup["perform_path_relinking"], 
                experiment_setup["time_until_adapt"], experiment_setup["generations_until_adapt"], experiment_setup["perform_lagrangian"], 
                experiment_setup["perform_adaptation"], experiment_setup["local_search_method"], experiment_setup["max_time_local_search"],
                experiment_setup["perform_local_search"], s_values):

            experiment = {
                "experiment_id": str(experiment_count),
                "execution_id": execution_id,
                "seed": int(experiment_setup["seed"]),
                "silent": bool(experiment_setup["silent"]),
                "instance": instance_name,
                "instance_path": instance_path,
                "plots_dir": plots_dir,
                "A": instance["A"],
                "R": instance["R"],
                "n": n,
                "m": m,
                "s": s,
                "best_known_result": float(best_known_result),
                "generate_plots": bool(experiment_setup["generate_plots"]),
                "max_generations": int(max_generations),
                "max_generations_without_change": int(experiment_setup["max_generations_without_change"]),
                "max_time": int(experiment_setup["max_time"]),
                "max_time_without_change": int(experiment_setup["max_time_without_change"]),
                "population_size": int(population_size),
                "encoding_method": encoding,
                "initialization_method": initialization,
                "selection_method": selection,
                "mutation_method": mutation,
                "self_mutation": bool(self_mutation),
                "crossover_method": crossover,
                "parent_selection_method": parent_selection,
                "mutation_probability": float(mutation_probability),
                "crossover_probability": float(crossover_probability),
                "perform_assexual_crossover": bool(assexual_crossover),
                "elite_size": float(elite_size),
                "offspring_size": float(offspring_size),
                "perform_path_relinking": bool(path_relinking),
                "avoid_clones": bool(experiment_setup["avoid_clones"]),
                "time_until_adapt": float(time_until_adapt),
                "generations_until_adapt": int(generations_until_adapt),
                "perform_lagrangian": bool(perform_lagrangian),
                "perform_adaptation": bool(perform_adaptation),
                "local_search_method": local_search_method,
                "max_time_local_search": float(max_time_local_search),
                "perform_local_search": bool(perform_local_search)
            }
            experiments.append(experiment)
            experiment_count += 1
            
    return experiments



def remove_experiments_to_be_ignored(experiment_setup, experiments):
    if not bool(experiment_setup["ignore_experiments_already_executed"]):
        print("No experiments to ignore.")
        return experiments

    results_file = os.path.join(experiment_setup["results_dir"], "results.csv")
    ignore_list_file = os.path.join(experiment_setup["results_dir"], "ignore_list.csv")
    ignore_list = None

    if os.path.exists(results_file):
        print("results.csv found.")
        ignore_list = pd.read_csv(results_file, header=0, sep=";")["experiment_id"].tolist()
    elif os.path.exists(ignore_list_file):
        print("ignore_list.csv found.")
        ignore_list = pd.read_csv(ignore_list_file, header=0, sep=";")["experiment_id"].tolist()
    else:
        return experiments

    new_experiments = []
    for experiment in experiments:
        if int(experiment["experiment_id"]) not in ignore_list:
            new_experiments.append(experiment)
        else:
            print("Ignoring experiment {}".format(experiment["experiment_id"]))

    return new_experiments


def run_experiment(experiment_setup, experiment):
    results = {}
    validated, message = validade_experiment_params(experiment)
    start_time = time.time()
    
    if validated:
        d_opt = GeneticAlgoritm(experiment)
        results, message = d_opt.loop()

    finish_time = time.time()
    
    with mutex:
        # Saves results to disk.
        save_results(
            experiment_setup, 
            experiment, 
            results, 
            start_time, 
            finish_time, 
            validated, 
            message
        )

def main(experiment_setup_file):
    experiment_setup = json.load(open(experiment_setup_file))
    experiments = build_experiments(experiment_setup)
    experiments = remove_experiments_to_be_ignored(experiment_setup, experiments)

    #sanitycheck = ["{}, {}".format(str(experiment["experiment_id"]), str(experiment["instance"])) for experiment in experiments]
    #for element in sanitycheck:
    #    print(element)

    print(f"Number of experiments: {len(experiments)}")
    if experiment_setup["num_processes"] > 1:
        pool = Pool(experiment_setup["num_processes"])
        # startmap is synchronous and, as such, will hold the execution of this called thread untill all maped jobs are executed.
        pool.starmap(run_experiment, zip(repeat(experiment_setup), experiments), chunksize=1) 
        pool.close()
    else:
        for experiment in experiments:
            run_experiment(experiment_setup, experiment)

    print("Finished.")

def save_results(experiment_setup, experiment, results, start_time, finish_time, validated, message):
    experiment_fields = ["experiment_id",
        "execution_id",
        "seed",
        "instance",
        "n",
        "m",
        "s",
        "max_generations",
        "max_generations_without_change",
        "max_time",
        "max_time_without_change",
        "population_size",
        "encoding_method",
        "initialization_method",
        "selection_method",
        "mutation_method",
        "self_mutation",
        "crossover_method",
        "parent_selection_method",
        "mutation_probability",
        "crossover_probability",
        "elite_size",
        "offspring_size",
        "perform_path_relinking",
        "generate_plots", # important to include because it affects the total time of an experiment.
        "avoid_clones",
        "perform_adaptation",
        "perform_local_search",
        "local_search_method",
        "best_known_result"
    ]

    results_fields = [
        # main results
        "best_solution_found",
        "gap",
        "num_generations",
        "total_time_ms",
        "start_time",
        "finish_time",
        "message",

        # ga loop results
        "ga_best_sol",
        "ga_best_sol_gap",
        "ga_total_time",

        # local search results
        "ls_improved_solution",
        "ls_best_sol",
        "ls_best_sol_gap",
        "ls_total_time",

        # path relinking results
        "pr_improved_solution",
        "pr_best_sol",
        "pr_best_sol_gap",
        "pr_total_time",

        # detailed results
        "best_sol_changes",
        "best_sol_change_times",
        "best_sol_change_generations",
        "elite_fitness",
        "ls_sol_values",
        "pr_sol_values",

        # hashes and raw solutions
        "raw_best_solution",
        "best_solution_hash",
        "elite_hashes",
        "improvement_candidates_hashes",
        "ls_individuals_hashes",
        "pr_individuals_hashes",
        
        # other
        "cpu_name",
        "version_hash"
    ]


    #sol_changes = {
    #    "best_sol_changes": [],
    #    "best_sol_change_times": [],
    #    "best_sol_change_generations": []
    #}
    
    #sol_changes = extra_results["sol_changes"] if "sol_changes" in extra_results else sol_changes
    #loop_start_time = extra_results["loop_start_time"] if "loop_start_time" in extra_results else "" # only used if sol_changes not empty
    #improvement_candidates = extra_results["improvement_candidates"] if "improvement_candidates" in extra_results else []
    #pr_individuals = extra_results["pr_individuals"] if "pr_individuals" in extra_results else []
    #pr_sol_values = extra_results["pr_sol_values"] if "pr_sol_values" in extra_results else []
    #ls_individuals = extra_results["ls_individuals"] if "ls_individuals" in extra_results else []
    #ls_sol_values = extra_results["ls_sol_values"] if "ls_sol_values" in extra_results else []

    
    results_file = os.path.join(experiment_setup["results_dir"], "results.csv")

    if not os.path.exists(results_file):
        with open(results_file, "w") as file:
            header = ";".join(experiment_fields + results_fields)
            file.write(header + "\n")

    with open(results_file, "a") as file:
            #experiment["seed"] = str(seed)

            #params_line = [experiment[field] for field in fields]
            #params_line = ";".join([str(value) for value in params_line])

            #elite_fitness = ",".join([str(individual.fitness) for individual in results])
            #elite_hash = ",".join([str(individual.individual_hash) for individual in results])

            #best_sol_changes = ",".join([str(element) for element in sol_changes["best_sol_changes"]])
            #best_sol_change_times = ",".join([str(element - loop_start_time) for element in sol_changes["best_sol_change_times"]])
            #best_sol_change_generations = ",".join([str(element) for element in sol_changes["best_sol_change_generations"]])

            #best_fitness = results[0].fitness if validated else ""
            #best_hash = results[0].individual_hash if validated else ""
            #raw_best_solution = "".join(str(int(gene)) for gene in results[0].binary_chromosome.T[0]) if validated else ""
            #raw_improvement_candidates = ",".join(["".join(str(int(gene)) for gene in individual.binary_chromosome.T[0]) for individual in improvement_candidates]) if validated else ""
            #pr_individuals = ",".join(["".join(str(int(gene)) for gene in individual.binary_chromosome.T[0]) for individual in pr_individuals]) if validated else ""
            #pr_sol_values = ",".join([str(value) for value in pr_sol_values]) if validated else ""
            #ls_individuals = ",".join(["".join(str(int(gene)) for gene in individual.binary_chromosome.T[0]) for individual in ls_individuals]) if validated else ""
            #ls_sol_values = ",".join([str(value) for value in ls_sol_values]) if validated else ""

            #gap = ""
            ##print(best_fitness, type(best_fitness))
            #if type(best_fitness) != str:
            #    #gap = (best_fitness - experiment["best_known_result"]) / experiment["best_known_result"]
            #    gap = (best_fitness - experiment["best_known_result"]) / abs(experiment["best_known_result"])


            version_hash = Utils.get_git_hash()

            cpu_name = ""
            try:
                cpu_name = cpuinfo.get_cpu_info()["brand_raw"]
            except:
                cpu_name = "Unable to retrieve CPU name."

            results.update({
                "start_time": datetime.fromtimestamp(start_time),
                "finish_time": datetime.fromtimestamp(finish_time),
                "total_time_ms": finish_time - start_time,
                "message": message,
                "cpu_name": cpu_name, 
                "version_hash": version_hash
            })


            results_line = []

            for field in experiment_fields + results_fields:
                value = ""
                if field in experiment_fields:
                    if field == "seed":
                        if "seed" in results:
                            value = str(results["seed"])
                        else:
                            value = str(experiment["seed"])
                    else:
                        value = str(experiment[field])

                elif field in results:
                    if type(results[field]) == list:
                        value = ",".join([str(element) for element in results[field]])
                    else:
                        value = str(results[field])    
                    
                results_line.append(value)

            results_line = ";".join(result for result in results_line)



            file.write(
                results_line + "\n"
            )
    
if __name__ == "__main__":
    experiment_setup_file = sys.argv[1]
    main(experiment_setup_file)
