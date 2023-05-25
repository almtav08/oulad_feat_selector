from typing import List
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from .genome import Genome
import time
import random
import numpy as np


class OEAlgorithm():
    def __init__(self, model: BaseEstimator, metric: str, shape: int, population_size: int = 200, mutation_rate: float = 0.2, survival_rate: float = 0.5, features_to_select: int = 1) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.survival_rate = survival_rate
        self.support = np.zeros(shape)
        self.model = model
        self.metric = metric
        self.features_to_select = features_to_select


# region Methods
    def fit(self, data: np.ndarray, target: np.ndarray, iteractions: int = 125) -> None:
        t0 = time.time()
        population: List['Genome'] = []
        killed: List['Genome'] = []

        for _ in range(self.population_size):
            genome = Genome(len(self.support), self.features_to_select)
            genome.random()
            population.append(genome)
            killed.append(genome)

        for _ in range(iteractions):
            for genome in killed:
                data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=42, stratify=target)
                model = clone(self.model)
                idxs = np.where(genome.get_support() == 1)[0]
                model.fit(data_train[:, idxs], target_train)
                pred = model.predict(data_test[:, idxs])
                genome.set_reward(accuracy_score(target_test, pred))

            killed.clear()
            population.sort(key=lambda g: g.get_reward(), reverse=True)
            first_killed_genome_index = int(self.survival_rate * self.population_size)
            for i in range(first_killed_genome_index, self.population_size):
                killed.append(population[i])

            for killed_genome in killed:
                parent_a_index = random.randrange(first_killed_genome_index)
                parent_b_index = random.randrange(first_killed_genome_index)
                while parent_a_index == parent_b_index and len(killed) > 1:
                    parent_b_index = random.randrange(first_killed_genome_index)

                killed_genome.crossover(population[parent_a_index], population[parent_b_index])

                if random.random() < self.mutation_rate:
                    killed_genome.mutate_at_random_index()

        self.support = population[0].get_support()

    def get_support(self) -> np.ndarray:
        return self.support
# endregion

# region Override
    def __str__(self):
        return f"OnlineEvolutionPlayer[{self.population_size!s}][{self.mutation_rate!s}][{self.survival_rate!s}]"
# endregion