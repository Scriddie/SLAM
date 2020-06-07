from genetic.ANN import ANN
from genetic.population import Population
from simulation.world_generator import WorldGenerator
from gui.ann_controller import apply_action, exponential_decay
import _experiments.visualize as visualize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
import subprocess
from pathlib import Path


# import multiprocessing as mp
# from itertools import repeat


class ANNCoverageEvaluator:
    def __init__(self, generator, input_dims, output_dims, hidden_dims,
                 feedback, eval_seconds, step_size_ms, feedback_time,
                 num_eval, normalization, world_name):
        self.generator = generator
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.eval_seconds = eval_seconds
        # According to the internetz humans have a reaction time of 200ms to 300ms, why does this matter?
        self.step_size_ms = step_size_ms
        self.feedback = feedback
        self.feedback_time = feedback_time
        self.num_eval = num_eval
        self.normalization = normalization
        self.world_name = world_name

    def generate_evaluate(self, genome, random_robot):
        world, robot = self.generator.create_world(random_robot=random_robot)
        return self.evaluate_in_world(world, robot, genome)

    def evaluate(self, genome):
        scores = []

        for step in range(10):
            scores.append(self.generate_evaluate(genome, True))

        return np.mean(scores)

        # Interesting result, not pooling is faster
        # setup multitreath poolling
        # threads = mp.cpu_count()
        # self.pool = mp.Pool(threads)
        #
        # scores = [self.pool.apply(self.generate_evaluate, args=(genome, True)) for step in range(10)]
        # print(scores)
        # pool.close()

    def evaluate_in_world(self, world, robot, genome):
        """
            Evaluate fitness of current genome (ANN weights)
        """
        ann = self.to_ann(genome)
        # Dirty Hack - Do an update to let the robot collect sensor data
        world.update(0)

        # We round up so that we'd rather overestimate evaluation time
        steps = int((self.eval_seconds * 1000) / self.step_size_ms) + 1
        delta_time = self.step_size_ms / 1000
        distance_sums = []
        for _ in range(steps):
            apply_action(robot, ann, self.feedback)
            world.update(delta_time)

            sensors = exponential_decay([dist for hit, dist in robot.sensor_data], start=100, end_factor=0.0, factor=1)
            distance_sums.append(np.sum(sensors))
        # If we hit negative values we are dead thus 0
        return world.dustgrid.cleaned_cells - np.sum(distance_sums)
        # return max(0, world.dustgrid.cleaned_cells - np.sum(distance_sums) * 20)  # 100

    def get_genome_size(self):
        genome_size = 0

        prev_dim = self.input_dims
        for i, hidden_dim in enumerate(self.hidden_dims):
            # for the last layer, add the weights for feedback
            if self.feedback:
                if i == len(self.hidden_dims) - 1:
                    genome_size += hidden_dim * hidden_dim
            genome_size += hidden_dim * (prev_dim + 1)
            prev_dim = hidden_dim

        genome_size += self.output_dims * (prev_dim + 1)
        return genome_size

    def to_ann(self, genome):
        """
            Initialize ANN manually with genome weights
            @param genome: flattened ANN weights
        """
        prev_dim = self.input_dims
        prev_index = 0

        weight_matrices = []
        for i, hidden_dim in enumerate(self.hidden_dims):

            # Generate weight matrix based on genome
            if i == (len(self.hidden_dims) - 1):  # feedback layer
                num_units = hidden_dim * (hidden_dim + prev_dim + 1)
                matrix = genome[prev_index:(prev_index + num_units)]
                matrix = matrix.reshape((hidden_dim, hidden_dim + prev_dim + 1))
            else:  # any other layer
                num_units = hidden_dim * (prev_dim + 1)
                # Generate the matrix based on the weights in the genome
                matrix = genome[prev_index:prev_index + num_units]
                matrix = matrix.reshape((hidden_dim, prev_dim + 1))

            weight_matrices.append(matrix)
            prev_dim = hidden_dim
            prev_index = prev_index + num_units

        # Generate the matrix for the output
        matrix = genome[prev_index:]
        matrix = matrix.reshape((self.output_dims, prev_dim + 1))
        weight_matrices.append(matrix)

        # Generate the ANN
        # ann = ANN(
        #     input_dims=self.input_dims,
        #     output_dims=self.output_dims,
        #     hidden_dims=self.hidden_dims,
        #     step_size_ms=self.step_size_ms,
        #     normalization=self.normalization,
        #     feedback_time=self.feedback_time)
        # ann.weight_matrices = weight_matrices
        # return ann

        # Generate the ANN
        ann = ANN(self.input_dims, self.output_dims, self.hidden_dims)
        ann.weight_matrices = weight_matrices
        return ann


def train(iterations, generator, evaluator, population, evaluator_args,
          population_args, world_name, save_modulo=50, experiment=""):
    max_fitness = []
    avg_fitness = []
    diversity = []
    if experiment == "":
        experiment = f"{datetime.now():%Y-%m-%d_%H-%S-%f}"
        experiment += "-" + world_name
    experiment = os.path.join("_experiments", experiment)
    if not os.path.isdir(experiment):
        os.mkdir(experiment)

    print("Start training experiment:", experiment)

    # save our model params
    with open(os.path.join(experiment, "population_args.txt"), "w+") as f:
        f.write(str(population_args))
    with open(os.path.join(experiment, "evaluator_args.txt"), "w+") as f:
        f.write(str(evaluator_args))

    for i in range(iterations):
        population.select(population_args["selection_rate"])
        population.crossover()
        population.mutate()

        # Collect some data
        fittest_genome = population.get_fittest_genome()
        max_fitness.append(population.get_max_fitness())
        avg_fitness.append(population.get_average_fitness())
        diversity.append(population.get_average_diversity())
        # # Early Stopping
        # if diversity[-1] < 0.02: #0.08
        #     # Save the best genome
        #     ann = evaluator.to_ann(fittest_genome['pos'])
        #     ann.save(f'./_checkpoints/model_{i}.p')
        #
        #     print("Early stopping due to low diversity")
        #     break

        # Print iteration data
        print(f"{i} - fitness:\t {evaluator.evaluate(fittest_genome['pos'])}")
        print(f"{i} - average fitness:\t {population.get_average_fitness()}")
        print("diversity:\t", population.get_average_diversity())
        if (i % save_modulo == 0) or (i == iterations - 1):
            # Save the best genome
            ann = evaluator.to_ann(fittest_genome['pos'])
            model_name = f"model_{i}"
            ann.save(os.path.join("_checkpoints", f"{model_name}.p"))
            ann.save(os.path.join(experiment, f"{model_name}.p"))
            # Take a snapshot of what robot outcomes look like
            subprocess.call(["python3", "main.py", "--snapshot",
                             "--snapshot_dir", f"{experiment}/{model_name}.png",
                             "--model_name", f"{model_name}.p", "--world_name", world_name])

    history = pd.DataFrame({
        "Max_Fitness": max_fitness,
        "Avg_Fitness": avg_fitness,
        "Diversity": diversity,
        "Iteration": [i for i in range(len(max_fitness))]
    })
    g = visualize.show_history(history, path=os.path.join(experiment, "ev_algo.png"))
    save_history(history, experiment)
    return ann, history


def save_history(history, experiment):
    timestamp = f"{datetime.now():%Y-%m-%d_%H-%S-%f}"
    file_name = os.path.join(experiment, f"{timestamp}.csv")
    history.to_csv(file_name)


if __name__ == "__main__":
    # Create folder for saving models
    Path("_checkpoints").mkdir(parents=True, exist_ok=True)

    # Initialization
    WIDTH = 400
    HEIGHT = 400
    POP_SIZE = 100
    FEEDBACK = True
    world_names = ["rect_world", "double_rect_world", "trapezoid_world", "double_trapezoid_world", "star_world",
                   "random"]
    world_num = 4
    world_num = 5
    world_name = world_names[world_num]

    robot_args = {
        "n_sensors": 12,
        "max_sensor_length": 100
    }
    generator = WorldGenerator(WIDTH, HEIGHT, 20, world_name)


    # Good until here
    evaluator_args = {
        "world_name": world_name,
        "generator": generator,
        "input_dims": robot_args["n_sensors"],
        "output_dims": 2,
        "hidden_dims": [16, 4],
        "feedback": FEEDBACK,
        "eval_seconds": 20,
        "step_size_ms": 270,  # 270
        "feedback_time": 270,  # 540
        "num_eval": 10,
        "normalization": robot_args["max_sensor_length"]
    }
    evaluator = ANNCoverageEvaluator(**evaluator_args)
    population_args = {
        "pop_size": POP_SIZE,
        "genome_size": evaluator.get_genome_size(),
        "eval_func": evaluator.evaluate,
        "init_func": np.random.normal,
        "weight_scale": 1.0,
        "mutation_rate": 0.1,
        "mutation_scale": 0.1,
        "selection_rate": 0.9
    }

    population = Population(
        pop_size=population_args["pop_size"],
        genome_size=population_args["genome_size"],
        eval_func=population_args["eval_func"],
        mutation_rate=population_args["mutation_rate"],
        mutation_scale=population_args["mutation_scale"],
        init_func=np.random.normal
    )

    # Train
    iterations = 200
    # ann, history = train(iterations=iterations, generator=generator, evaluator=evaluator, population=population)
    ann, history = train(
        iterations=iterations,
        generator=generator,
        evaluator=evaluator,
        population=population,
        world_name=world_name,
        evaluator_args=evaluator_args,
        population_args=population_args
    )

    g = visualize.show_history(history)
    plt.show()

    ann.show()