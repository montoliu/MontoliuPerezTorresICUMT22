# ---------------------------------------
# The Knn Fitness
# ---------------------------------------
# k, distance_function, data_representation, weights
import numpy as np
from KnnClassifier import KnnClassifier


class KnnFitness:
    def __init__(self, fps_train, loc_train, fps_test, loc_test):
        self.fps_train = fps_train
        self.fps_test = fps_test
        self.loc_train = loc_train
        self.loc_test = loc_test

        self.min_dataset = np.min(self.fps_train)
        self.not_detected = 100

        self.cl = KnnClassifier(fps_train, loc_train, fps_test, loc_test)

    def evaluate(self, parameters):
        parameters = self.ntbea2knn(parameters)
        loc_estimated = self.cl.get_loc_test_samples(parameters)
        mean_error = self.cl.estimate_mean_error(loc_estimated, self.loc_test)
        if mean_error > 25:
            mean_error = 25
        return 25 - mean_error

    def ntbea2knn(self, ntbea_parameters):
        knn_parameters = [0, 0, 0, 0]
        knn_parameters[0] = self.get_knn_from_parameters(ntbea_parameters[0])
        knn_parameters[1] = self.get_dist_function_from_parameters(ntbea_parameters[1])
        knn_parameters[2] = self.get_data_representation_from_parameters(ntbea_parameters[2])
        knn_parameters[3] = self.get_weights_from_parameters(ntbea_parameters[3])

        return knn_parameters

    def get_knn_from_parameters(self, parameter):
        if parameter == 0:
            return 1
        elif parameter == 1:
            return 3
        elif parameter == 2:
            return 5
        elif parameter == 3:
            return 7
        elif parameter == 4:
            return 9
        elif parameter == 5:
            return 11
        elif parameter == 6:
            return 13
        elif parameter == 7:
            return 15
        elif parameter == 8:
            return 17
        elif parameter == 9:
            return 19

    def get_dist_function_from_parameters(self, parameter):
        if parameter == 0:
            return 'euclidean'
        elif parameter == 1:
            return 'city_block'
        elif parameter == 2:
            return 'sorensen'
        elif parameter == 3:
            return 'neyman'

    def get_data_representation_from_parameters(self, parameter):
        if parameter == 0:
            return 'positive'
        elif parameter == 1:
            return 'normalized'
        elif parameter == 2:
            return 'exponential'
        elif parameter == 3:
            return 'powed'

    def get_weights_from_parameters(self, parameter):
        return parameter