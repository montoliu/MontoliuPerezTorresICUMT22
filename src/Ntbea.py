import copy
import math
import random
import time
import statistics
from Bandit1D import Bandit1D
from Bandit2D import Bandit2D


# --------------------------------------------------------------
# NTBEA Algorithm
# --------------------------------------------------------------
class Ntbea:
    def __init__(self, l_dimensions, c, fitness, n_neighbours, mutation_probability):
        self.C = c  # C parameter for ucb
        self.n_neighbours = n_neighbours  # Number of neighbours in ntbea iteration
        self.mutation_probability = mutation_probability  # Mutation probability in the mutate step
        self.l_bandits1D = []  # 1D bandits
        self.l_bandits2D = []  # 2D bandits
        self.n_dimensions = len(l_dimensions)  # Dimension of the problem (i.e. number of parameters)

        self.l_dimensions = l_dimensions  # List of number of possible values of each parameter
        self.n_bandits1D = self.n_dimensions  # Number of 1D bandits
        self.n_bandits2D = (self.n_dimensions * (self.n_dimensions - 1)) / 2  # Number of 2D bandits

        self.create_bandits()  # Initialize the 1D and 2D bandits
        self.fitness = fitness  # Fitness function
        self.l_currents = []  # List of the selected individuals

        self.verbose = False

    # --------------------------------------------------------------------
    # Initialize the bandits
    # --------------------------------------------------------------------
    def create_bandits(self):
        # Create empty 1D bandits
        for i in range(self.n_dimensions):
            new_bandit = Bandit1D(self.C)
            self.l_bandits1D.append(new_bandit)

        # Create empty 2D bandits
        for i in range(0, self.n_dimensions - 1):
            for j in range(i + 1, self.n_dimensions):
                new_bandit = Bandit2D(self.C)
                self.l_bandits2D.append(new_bandit)

    # --------------------------------------------------------------------
    # Run the NTBEA algorithm
    # --------------------------------------------------------------------
    def run(self, n_iterations, n_initialization, verbose=False):
        self.verbose = verbose
        current, score = self.initialization(n_initialization)
        self.l_currents.append(current)              # Append to the list
        if self.verbose:
            print(str(current) + " " + str(10 - score))

        iteration = 1
        while iteration < n_iterations:
            population = self.get_neighbours(current, self.n_neighbours, self.mutation_probability)  # get neigbours
            current = self.get_best_individual(population)  # Get best neighbour using bandits
            score = self.fitness.evaluate(current)          # Get the score of the new current individual
            if self.verbose:
                print(str(current) + " " + str(10-score))

            self.l_currents.append(current)      # Append to the list
            self.update_bandits(current, score)  # Update bandits
            iteration += 1

        return self.recommend_solution()

    # --------------------------------------------------------------------
    # Fill the ntbea table with some initial individuals
    # --------------------------------------------------------------------
    def initialization(self, n_initialization):
        population = self.get_random_population(n_initialization)

        best_individual = None
        best_score = -math.inf

        for individual in population:
            score = self.fitness.evaluate(individual)  # Get the score of the current individual
            self.update_bandits(individual, score)  # Update bandits
            if self.verbose:
                print("Initialization with: " + str(individual) + " " + str(10 - score))
            if score > best_score:
                best_score = score
                best_individual = individual

        return best_individual, best_score

    # --------------------------------------------------------------------
    # Get a random population
    # --------------------------------------------------------------------
    def get_random_population(self, n_neighbours):
        population = []
        while len(population) < n_neighbours:
            individual = self.get_random_individual()
            if not (individual in population):
                population.append(individual)
        return population

    # --------------------------------------------------------------------
    # Return a random individual
    # --------------------------------------------------------------------
    def get_random_individual(self):
        individual = []
        for i in range(self.n_dimensions):
            n = random.randint(0, self.l_dimensions[i] - 1)
            individual.append(n)
        return individual

    # --------------------------------------------------------------------
    #  Update the bandits
    # --------------------------------------------------------------------
    # Given and individual (i.e. [1,3,4]) update all the bandits
    def update_bandits(self, individual, score):
        # 1D
        for i in range(self.n_bandits1D):
            element = individual[i]
            self.l_bandits1D[i].update(element, score)

        # 2D
        k = 0
        for i in range(0, self.n_dimensions - 1):
            for j in range(i + 1, self.n_dimensions):
                element1 = individual[i]
                element2 = individual[j]
                self.l_bandits2D[k].update(element1, element2, score)
                k += 1

    # --------------------------------------------------------------------
    # Returns the mean of all ucb of each bandit
    # An element not in a bandit returns a big number
    # --------------------------------------------------------------------
    def get_total_ucb(self, individual):
        acm = 0

        # 1D
        for i in range(0, self.n_dimensions):
            element = individual[i]
            acm += self.l_bandits1D[i].ucb(element)
            i += 1

        # 2D
        k = 0
        for i in range(0, self.n_dimensions - 1):
            for j in range(i + 1, self.n_dimensions):
                element1 = individual[i]
                element2 = individual[j]
                acm += self.l_bandits2D[k].ucb(element1, element2)
                k += 1

        return acm / (self.n_bandits1D + self.n_bandits2D)

    # --------------------------------------------------------------------
    # Returns the mean of all ucb of each bandit.
    # An element not in a bandit returns 0
    # --------------------------------------------------------------------
    def get_total_ucb_final(self, individual):
        acm = 0

        # 1D
        for i in range(0, self.n_dimensions):
            element = individual[i]
            acm += self.l_bandits1D[i].ucb_final(element)
            i += 1

        # 2D
        k = 0
        for i in range(0, self.n_dimensions - 1):
            for j in range(i + 1, self.n_dimensions):
                element1 = individual[i]
                element2 = individual[j]
                acm += self.l_bandits2D[k].ucb_final(element1, element2)
                k += 1

        return acm / (self.n_bandits1D + self.n_bandits2D)

    # --------------------------------------------------------------------
    # Obtain n_neighbours from an individual
    # Change at least one parameter (randomly chosen).
    # The rest can be changed depending of the mutation probability
    # --------------------------------------------------------------------
    def get_neighbours(self, individual, n_neighbours, mutation_probability):
        population = []
        while len(population) < n_neighbours:
            neighbour = copy.copy(individual)
            i = random.randint(0, self.n_dimensions - 1)  # the parameter to be changed
            for j in range(self.n_dimensions):
                if i == j:  # The parameter chosen is always mutated
                    self.mutate_gen(neighbour, j)
                else:  # The rest can be mutated depending of the mutation prob.
                    n = random.random()
                    if n < mutation_probability:
                        self.mutate_gen(neighbour, j)
            if not neighbour in population:
                if not neighbour in self.l_currents:
                    population.append(neighbour)
        return population

    def get_final_neighbours(self, individual, n_neighbours):
        population = []
        for i in range(n_neighbours):
            neighbour = copy.copy(individual)
            i = random.randint(0, self.n_dimensions - 1)  # the parameter to be changed
            self.mutate_gen(neighbour, i)
            if not neighbour in population:
                population.append(neighbour)
        return population

    # --------------------------------------------------------------------
    # Mutate the j-th gen of an individual
    # The mutation consists of change the value of the j-th gen using a different valid one
    # --------------------------------------------------------------------
    def mutate_gen(self, individual, j):
        prev_value = individual[j]
        new_value = random.randint(0, self.l_dimensions[j] - 1)
        while new_value == prev_value:  # if it is the same, try again
            new_value = random.randint(0, self.l_dimensions[j] - 1)

        individual[j] = new_value

    # --------------------------------------------------------------------
    # Get best individual from a population. It is the one with greater ucb
    # --------------------------------------------------------------------
    def get_best_individual(self, population):
        best_ucb = -math.inf
        best_individual = population[0]

        for individual in population:
            ucb = self.get_total_ucb(individual)
            if ucb > best_ucb:
                best_ucb = ucb
                best_individual = individual

        return best_individual

    # --------------------------------------------------------------------
    # Recommend solution
    # It is the one, included in l_currents and its neighbours, with best total ucb
    # Note that get_total_ucb_final is used instead of get_total_ucb
    # i.e. when some element (or pair of elements) are not in the bandits, return 0.0 instead of a big number
    # --------------------------------------------------------------------
    def recommend_solution(self):
        final_population = []

        # get neighbours of the individuals form l_currents
        for current in self.l_currents:
            population = self.get_final_neighbours(current, self.n_neighbours)
            final_population.append(current)
            final_population.extend(population)

        # get the best individual
        best_score = -math.inf
        best_individual = None
        for individual in final_population:
            score = self.get_total_ucb_final(individual)
            if score > best_score:
                best_individual = individual
                best_score = score

        return best_individual
