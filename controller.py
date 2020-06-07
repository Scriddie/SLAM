import numpy as np
from genetic.functions import rastrigin, rosenbrock
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import genetic.plots as plots
import os

from genetic.population import Population


def show(pop):
    for j, i in enumerate(pop.individuals):
        if j % 10 == 0:
            x1 = i["pos"][0]
            x2 = i["pos"][1]
            fitness = i["fitness"]
            print(f"{x1:.2f} {x2:.2f}\tfit {fitness:.2f}")
    avg_fitness = np.mean([i["fitness"] for i in pop.individuals])
    print(f"average fitness: {avg_fitness}\n")


def evolve(n, pop):
    """train function"""
    history = []
    for i in range(n):
        pop.select(0.90)
        pop.crossover()
        pop.mutate()
        history.append(deepcopy(pop.individuals))
        if i % 100 == 0:
            show(pop)
    return history


def process_history(history):
    """create a df from history for plotting"""
    fitness = []
    diversity = []
    for pop in history:
        fitness.append(np.mean([i["fitness"] for i in pop]))
        x1_diversity = np.var([i["pos"][0] for i in pop])
        x2_diversity = np.var([i["pos"][1] for i in pop])
        div = np.linalg.norm([x1_diversity, x2_diversity])
        diversity.append(div)

    df = pd.DataFrame({
        "fitness": fitness,
        "diversity": diversity
    })
    df["Generation"] = df.index
    df = pd.melt(df, id_vars=['Generation'],
        value_vars=["fitness", "diversity"])
    return df


if __name__ == "__main__":
    functions = [rosenbrock, rastrigin]
    function_num = 1
    fn = functions[function_num]
    population = Population(100, 2, lambda x: -fn(x), mutation_scale=0.2, init_func=lambda size: np.random.uniform(low=-2,high=2,size=size))
        
    history = evolve(100, population)

    df = process_history(history)
    g = sns.FacetGrid(data=df, row="variable", height=3, sharey="row")
    g = g.map(plt.plot, "Generation", "value")
    # sns.lineplot(data=df, x="Generation", y="value", hue="variable")
    plt.show()

    extent = [-2, 2, -2, 2]
    file_name = "ae"
    img_dir = ""
    if(function_num == 0):
        file_name = "ae_rosenbrock"
    elif(function_num == 1):
        file_name = "ae_rastrigin"
    plots.visualize_heatmap(fn, history, extent, trail_lenght=5,
        fname=os.path.join(img_dir, file_name+".gif"), output="step")
