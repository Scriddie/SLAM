import numpy as np
from copy import deepcopy

# • Devise genetic representation
# • Build a population
# • Design a fitness function
# • Choose selection method
# • Choose crossover & mutation
# • Choose data analysis method

class Population:
    def __init__(self, pop_size, genome_size, eval_func,
                 mutation_rate=0.1, mutation_scale=0.2,
                 init_func=np.random.uniform):
        self.pop_size = pop_size
        self.genome_size = genome_size
        self.eval_func = eval_func
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.init_func = init_func
        self.individuals = []
        
        self.__generate_population__()
        
    def select(self, percentage):
        """
            1) Sort individuals by fitness, divide by squared number of individuals
            2) Select <percentage> according to number obtained in 1)
        """
        sorted_individuals = sorted(self.individuals, key=lambda x: x["fitness"])
        total_rank_sum = len(self.individuals) * (len(self.individuals) + 1) / 2
        probabilities = [(i+1)/total_rank_sum for i in range(len(sorted_individuals))]
        num = int(len(sorted_individuals) * percentage)
        self.individuals = np.random.choice(sorted_individuals, num, p=probabilities).tolist()
        
    def regenerate(self):
        # TODO: why not replicate / stay close to the current population?
        """
            Generates random genomes until the population is full
        """
        for _ in range(self.pop_size - len(self.individuals)):
            self.individuals.append(self.__generate_child__())
        
    def crossover(self):
        """
            Generate some new individuals that are a mix of existing ones
        """
        num_crossovers = (self.pop_size - len(self.individuals)) // 2
        for _ in range(num_crossovers):
            # choose two particles for mutation
            p1 = self.individuals[np.random.randint(0, len(self.individuals))]
            p2 = self.individuals[np.random.randint(0, len(self.individuals))]
            # combine genes at "cross over point"
            crossover_point = np.random.randint(1, self.genome_size)
            c1_genome = np.array([*p1["pos"][:crossover_point],
                                  *p2["pos"][crossover_point:]])
            c2_genome = np.array([*p2["pos"][:crossover_point], 
                                  *p1["pos"][crossover_point:]])
            # create new individuals from resulting genes
            c1 = {"pos": c1_genome, "fitness": self.eval_func(c1_genome)}
            c2 = {"pos": c2_genome, "fitness": self.eval_func(c2_genome)}                
            self.individuals.append(c1)
            self.individuals.append(c2)
                    
    def mutate(self):
        """
            mutate some random genomes
        """
        for p in self.individuals:
            if np.random.uniform(0, 1) < self.mutation_rate:
                p["pos"] += np.random.normal(scale=self.mutation_scale,
                    size=self.genome_size)
                p["fitness"] = self.eval_func(p["pos"])
                
    def get_fittest_genome(self):
        return max(self.individuals, key=lambda x: x["fitness"])
    
    def get_max_fitness(self):
        return self.get_fittest_genome()["fitness"]
    
    def get_average_fitness(self):
        return np.mean([indiv["fitness"] for indiv in self.individuals])
    
    def get_average_diversity(self):
        diversities = [self.get_diversity(indiv) for indiv in self.individuals]
        return np.mean(diversities)
            
    def get_diversity(self, genome):
        distance_sum = 0
        for indiv in self.individuals:
            distance_sum += np.sqrt((indiv["pos"] - genome["pos"]) ** 2)
            
        return distance_sum / len(self.individuals)

    def show(self):
        for i in self.individuals:
            print(f"{i}\n")
        
    def __generate_population__(self):
        self.individuals = []
        for i in range(self.pop_size):
            ind = self.__generate_child__()
            self.individuals.append(ind)
            
    def __generate_child__(self):
        x = self.init_func(size=self.genome_size)
        fitness = self.eval_func(x)
        ind = {"pos": x, "fitness": fitness}
        return ind