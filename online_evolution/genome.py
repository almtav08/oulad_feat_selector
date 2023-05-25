import random
import numpy as np

class Genome():
    def __init__(self, input_shape: int, number_of_features: int = 1):
        self.support = np.zeros(input_shape)
        self.reward = 0.0
        self.number_of_features = number_of_features

# region Methods
    def random(self) -> None:
        if self.number_of_features > len(self.get_support()):
            raise ValueError("Number of features cannot be greater than input shape")
        support = np.zeros(len(self.get_support()))
        idxs = random.sample(range(len(self.get_support())), self.number_of_features)
        support[idxs] = 1
        self.support = support

    def crossover(self, parent_a: 'Genome', parent_b: 'Genome') -> None:
        self.reward = 0.0
        for i in range(len(self.get_support())):
            if bool(random.getrandbits(1)):
                if (len(parent_a.get_support()) >= i):
                    self.get_support()[i] = parent_a.get_support()[i]
                elif (len(parent_b.get_support()) >= i):
                    self.get_support()[i] = parent_b.get_support()[i]
            else:
                if (len(parent_b.get_support()) >= i):
                    self.get_support()[i] = parent_b.get_support()[i]
                elif (len(parent_a.get_support()) >= i):
                    self.get_support()[i] = parent_a.get_support()[i]
        self.fix_support()
    
    def mutate_at_random_index(self) -> int:
        mutation_index = random.randrange(len(self.get_support()))
        for i in range(len(self.get_support())):
            if i == mutation_index:
                self.get_support()[i] = int(not self.get_support()[i])
        self.fix_support(mutation_index)

    def fix_support(self, mutation_index: int = -1) -> None:
        if (int(np.sum(self.get_support())) < int(self.number_of_features)):
            for _ in range(int(self.number_of_features) - int(np.sum(self.get_support()))):
                idxs = self.get_unsupported_idxs(mutation_index)
                idx = idxs[random.choice(range(idxs.shape[0]))]
                self.get_support()[idx] = 0
        if (int(np.sum(self.get_support())) > int(self.number_of_features)):
            for _ in range(int(np.sum(self.get_support())) - int(self.number_of_features)):
                idxs = self.get_supported_idxs(mutation_index)
                idx = idxs[random.choice(range(idxs.shape[0]))]
                self.get_support()[idx] = 0

    def get_supported_idxs(self, mutation_index: int) -> np.ndarray:
        support = np.where(self.get_support() == 1)[0]
        return np.delete(support, np.where(support == mutation_index)[0])
    
    def get_unsupported_idxs(self, mutation_index: int) -> np.ndarray:
        support = np.where(self.get_support() == 0)[0]
        return np.delete(support, np.where(support == mutation_index)[0])
                
        
# endregion

# region Getters
    def get_support(self) -> np.ndarray:
        return self.support

    def get_reward(self) -> float:
        return self.reward

    def set_reward(self, reward: float) -> None:
        self.reward = reward
# endregion

# region Override
    def __str__(self):
        return f"Genome [support={self.support}, reward={self.reward}]"
# endregion