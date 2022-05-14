import random
import math


# --------------------------------------------------------------
# Bandit1D class
# --------------------------------------------------------------
class Bandit1D:
    def __init__(self, c):
        self.C = c            # C parameter of the UCB equation
        self.score = dict()   # Mean score of each element of the bandit
        self.n = dict()       # Number of times that each element of the bandit has been accesed
        self.n_total = 0      # Number of times that the bandit has been accessed

    # Returns the score of an existing element
    def get_score(self, element):
        return self.score[element]

    # Updates the bandit with an element and its score
    # If not exist yet, it is added
    def update(self, element, score):
        if element in self.score:
            self.score[element] = (self.score[element] * self.n[element] + score) / (self.n[element] + 1)
            self.n[element] += 1
        else:
            self.score[element] = score
            self.n[element] = 1
        self.n_total += 1

    # Returns the ucb value for an element
    def ucb(self, element):
        if element in self.score:
            return self.score[element] + self.C * math.sqrt(math.log(self.n_total) / self.n[element])
        else:
            return 10e6 + random.random()  # If the element is not in the bandit, returns a random big number

    # This is the same than ucb but, in this case, when the element is not in the bandit returns 0.0
    def ucb_final(self, element):
        if element in self.score:
            return self.score[element] + self.C * math.sqrt(math.log(self.n_total) / self.n[element])
        else:
            return 0

    # Returns the element with the greater score
    def get_element_best_score(self):
        best_element = 0
        best_score = 0
        for element in self.score:
            if self.score[element] > best_score:
                best_score = self.score[element]
                best_element = int(element)
        return best_element

    # Returns the element with the greater ucb
    def get_element_best_ucb(self):
        best_element = 0
        best_ucb = 0
        for element in self.score:
            ucb_value = self.ucb(element)
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_element = int(element)
        return best_element

    def __repr__(self):
        return str(self.score) + " " + str(self.n)
