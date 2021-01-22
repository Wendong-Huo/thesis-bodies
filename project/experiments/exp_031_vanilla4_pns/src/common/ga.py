# Algo modified from pseudo code: https://leap-gmu.readthedocs.io/en/latest/distributed.html#asynchronous-fitness-evaluations
# There will be no master process looping forever.
# Evoke master code after every training process finishs.

# make a sqlite database file
# record P_p in database
# record configuration in database
# start a training
# at the end of training, evoke master (be careful about racing condition. sleep if others are working. won't take too long.)
# master read P_p and compare, reproduce, submit new jobs.

import os
import time
import yaml
import numpy as np
import copy

from common import seeds
from common import utils

class GA:
    database_path = "output_data/ga/ga.yml"
    submit_path = "22.1.submit_ga.sh"
    train_path = "22.2.train_wrapper.py"

    def __init__(self):
        # self.log = utils.Log(f"{self.database_path}.log")
        self.log = utils.Log()
        self.is_in_critical_section = False

    def create_db(self, force=False):
        data = {
            "setting": {
                "size_population_eval": 10,     # P_o in Algo 1.
                "size_population_parent": 10,   # P_p in Algo 1.
                "num_eval": 0,
                "latest_id": -1,
                "tournament_k": 5,
            },
            "population_parent": []
        }
        if not force:
            assert not os.path.exists(self.database_path), f"{self.database_path} already exists."
        with open(self.database_path, "w") as f:
            yaml.dump(data, f, Dumper=yaml.SafeDumper)

    def get_num_eval(self, data):
        # use command: squeue -n exp_name | wc -l
        # assert it's the same

        # for debug
        return data["setting"]["num_eval"]
    
    def ga_select_parent(self, population, k = 5):
        # Select two parents
        # Tournament Selection: https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
        # k-Way tournament
        # small k means less selection pressure and more randomness
        with seeds.temp_seed(int(time.time())):
            selected = np.random.randint(low=0, high=len(population), size=[k])
        fitnesses = []
        for i in range(k):
            fitnesses.append(population[selected[i]]["fitness"])
        best = np.argsort(fitnesses)[::-1]
        self.log.record(f"best: {selected[best]}")
        return population[selected[best[0]]]

    def ga_mutate(self, parent, n=2):
        # Mutate individual
        # Swap Mutation: https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm
        # small n means less mutation and less randomness
        offspring = copy.deepcopy(parent)
        offspring["parent"] = parent["id"]

        with seeds.temp_seed(int(time.time())):
            target_sequence_ids = np.random.randint(low=0, high=len(offspring["geno"]), size=[n])
            target_positions = []
            for i in range(n):
                target_pos = np.random.choice(np.arange(start=0, stop=len(offspring["geno"][0])), size=2)
                target_positions.append(target_pos)
        for i in range(n):
            offspring["geno"][target_sequence_ids[i]][target_positions[i][0]], offspring["geno"][target_sequence_ids[i]][target_positions[i][1]] = \
                offspring["geno"][target_sequence_ids[i]][target_positions[i][1]], offspring["geno"][target_sequence_ids[i]][target_positions[i][0]]
        self.log.record(f"parent geno    {parent['geno']}")
        self.log.record(f"offspring geno {offspring['geno']}")
        return offspring

    def checkout(self, data):
        data["setting"]["num_eval"] -= 1

    def slurm_submit(self, ind, data):
        # Submit offspring
        # use command: sbatch -J <exp_name> {self.submit_path} python {self.train_path} <filename>

        self.log.record(f"submit individual {ind['id']} whose parents are {ind['parent']}")
        data["setting"]["num_eval"] += 1
        self.log.record(f"in total, {data['setting']['num_eval']} jobs running")
        return

    def evoke_master(self, individual):
        # create lock
        while os.path.exists(f"{self.database_path}.lock"):
            with seeds.temp_seed(int(time.time())):
                l = np.random.randint(low=1, high=30)
                self.log.record(f"sleep for {l} sec...")
                time.sleep(l)
        open(f"{self.database_path}.lock", "w").close()
        # !! entering critical section
        self.is_in_critical_section = True

        data = self._evoke_master( individual )

        self.is_in_critical_section = True
        # !! leaving critical section
        os.remove(f"{self.database_path}.lock")
        return data

    def _evoke_master(self, individual):
        with open(self.database_path, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        
        self.checkout(data)
        if data["setting"]["size_population_parent"] > len(data["population_parent"]):
            # Pool not full yet, so just insert
            self.log.record(f"Insert {individual['id']} to population_parent")
            data["population_parent"].append(individual)
            self.log.record(f"current population_parent size {len(data['population_parent'])}")
        else:
            # Replace only if better than weakest in pool
            min_fitness = np.inf
            min_idx = -1
            for idx, p in enumerate(data["population_parent"]):
                if min_fitness> p["fitness"]:
                    min_idx = idx
                    min_fitness = p["fitness"]
            if individual["fitness"] > min_fitness:
                self.log.record(f"Replace {data['population_parent'][min_idx]['id']} with {individual['id']} in population_parent")
                data["population_parent"][min_idx] = individual
            else:
                self.log.record(f"Discard {individual['id']}")
        num_offspring_needed = data["setting"]["size_population_eval"] - self.get_num_eval(data)
        for i in range(num_offspring_needed):
            # Select parents
            parent = self.ga_select_parent(data["population_parent"])
            # mutate
            offspring = self.ga_mutate(parent)
            data["setting"]["latest_id"] += 1
            offspring["id"] = data["setting"]["latest_id"]
            # submit
            self.log.record(f"Submit offspring {offspring['id']}")
            self.slurm_submit(offspring, data)
            pass
        
        with open(self.database_path, "w") as f:
            yaml.dump(data, f, Dumper=yaml.SafeDumper)
        self.log.dump()
        return data

    def force_clean_lock(self):
        if os.path.exists(f"{self.database_path}.lock"):
            os.remove(f"{self.database_path}.lock")
