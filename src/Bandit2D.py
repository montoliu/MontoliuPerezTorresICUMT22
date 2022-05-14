import random
import math


# --------------------------------------------------------------
# Bandit class
# --------------------------------------------------------------
class Bandit2D:
    def __init__(self, c):
        self.C = c           # C parameter of the UCB equation
        self.score = dict()  # Mean score of each element of the bandit
        self.n = dict()      # Number of times that each element of the bandit has been accesed
        self.n_total = 0     # Number of times that the bandit has been accessed
        self.factor = 1000   # A pair of elements [e1,e2] will be transformed to e1*factor+e2 to be an unique element

    # Returns the score of an existing pair of elements
    def get_score(self, element1, element2):
        element = self.get_element(element1, element2)
        return self.score[element]

    # Updates the bandit with a pair of elements and its score
    # If not exist yet, it is added
    def update(self, element1, element2, score):
        element = self.get_element(element1, element2)   # From two elements to just one
        if element in self.score:
            self.score[element] = (self.score[element] * self.n[element] + score) / (self.n[element] + 1)
            self.n[element] += 1
        else:
            self.score[element] = score
            self.n[element] = 1
        self.n_total += 1

    # Returns the ucb value for a pair of elements
    def ucb(self, element1, element2):
        element = self.get_element(element1, element2)
        if element in self.score:
            return self.score[element] + self.C * math.sqrt(math.log(self.n_total) / self.n[element])
        else:
            return 10e6 + random.random()  # If the element is not in the bandit, returns a random big number

    # This is the same than ucb but, in this case, when the element is not in the bandit returns 0.0
    def ucb_final(self, element1, element2):
        element = self.get_element(element1, element2)
        if element in self.score:
            return self.score[element] + self.C * math.sqrt(math.log(self.n_total) / self.n[element])
        else:
            return 0

    # Returns the pair of elements with the greater score
    def get_elements_best_score(self):
        best_element = 0
        best_score = 0
        for element in self.score:
            if self.score[element] > best_score:
                best_score = self.score[element]
                best_element = int(element)
        element1, element2 = self.get_elements(best_element)
        return element1, element2

    # Returns the pair of elements with the greater ucb
    def get_elements_best_ucb(self):
        best_element = 0
        best_ucb = 0
        for element in self.score:
            element1, element2 = self.get_elements(element)
            ucb_value = self.ucb(element1, element2)
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_element = int(element)
        element1, element2 = self.get_elements(best_element)
        return element1, element2

    # From a pair of elements to the unique one
    # e.g [2,3] will be tranformed to 2*factor + 3
    def get_element(self, element1, element2):
        return element1 * self.factor + element2

    # From an elemento to a pair of them
    # e.g. if factor is 1000, an element 2005, will be tranformed in the pair [2,5]
    def get_elements(self, element):
        element1 = element // self.factor
        element2 = element % self.factor
        return element1, element2

    def __repr__(self):
        return str(self.score) + " " + str(self.n)
