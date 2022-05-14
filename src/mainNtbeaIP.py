import math
import statistics
import time

import numpy as np

from dataset import read_dataset
from Ntbea import Ntbea
from KnnFitness import KnnFitness


def vote_for_best_solution(l_solutions, l_dimensions):
    votes_k = np.zeros(l_dimensions[0])
    votes_dist = np.zeros(l_dimensions[1])
    votes_rep = np.zeros(l_dimensions[2])
    votes_wei = np.zeros(l_dimensions[3])

    for solution in l_solutions:
        votes_k[solution[0]] += 1
        votes_dist[solution[1]] += 1
        votes_rep[solution[2]] += 1
        votes_wei[solution[3]] += 1

    best_solution = ""
    best_k = np.argmax(votes_k) * 2 + 1
    best_solution += str(best_k)

    best_dist = np.argmax(votes_dist)
    if best_dist == 0:
        best_solution += ", euclidean"
    elif best_dist == 1:
        best_solution += ", city_block"
    elif best_dist == 2:
        best_solution += ", sorensen"
    elif best_dist == 3:
        best_solution += ", neyman"

    best_rep = np.argmax(votes_rep)
    if best_rep == 0:
        best_solution += ", positive"
    elif best_rep == 1:
        best_solution += ", normalized"
    elif best_rep == 2:
        best_solution += ", exponential"
    elif best_rep == 3:
        best_solution += ", powed"

    best_w = np.argmax(votes_wei)
    best_solution += ", " + str(best_w)

    return best_solution


def do_dataset(dataset):
    fps_train, loc_train, fps_test, loc_test = read_dataset(dataset)
    print("DATASET = " + dataset)
    # NTBEA parameters
    l_dimensions = [10, 4, 4, 3]
    c = math.sqrt(2)
    n_neighbours = 50
    mutation_probability = 0.2
    n_iterations = 10
    n_initialization = 20
    fitness = KnnFitness(fps_train, loc_train, fps_test, loc_test)

    # repeat 10 times
    l_solutions = []
    l_scores = []
    l_times = []
    out_str = ""
    for i in range(10):
        ntbea = Ntbea(l_dimensions, c, fitness, n_neighbours, mutation_probability)
        t0 = time.process_time()
        solution = ntbea.run(n_iterations, n_initialization, verbose=False)
        score = 25-fitness.evaluate(solution)
        t = time.process_time() - t0
        print("Solution: " + str(solution) + " " + str(score) + " " + str(t))
        out_str += "Solution: " + str(solution) + " " + str(score) + " " + str(t) + "\n"
        l_scores.append(score)
        l_solutions.append(solution)
        l_times.append(t)

    best_solution = vote_for_best_solution(l_solutions, l_dimensions)
    out_str += "Most voted solution: " + str(best_solution) + "\n"
    out_str += "Mean error         : " + str(statistics.mean(l_scores)) + "\n"
    out_str += "Std error          : " + str(statistics.stdev(l_scores)) + "\n"
    out_str += "Mean time          : " + str(statistics.mean(l_times)) + "\n"

    out_filename = "../out/" + dataset + "_ntbea_20_10_50_02.txt"
    out_file = open(out_filename, "w")
    out_file.write(out_str)
    out_file.close()


# ----------------------------------------
# MAIN
# ----------------------------------------
if __name__ == "__main__":
    do_dataset("DSI1")
    do_dataset("DSI2")
    do_dataset("MAN1")
    do_dataset("MAN2")
    do_dataset("TUT1")
    do_dataset("TUT2")
    do_dataset("UEXB1")
    do_dataset("UEXB2")
    do_dataset("UEXB3")
