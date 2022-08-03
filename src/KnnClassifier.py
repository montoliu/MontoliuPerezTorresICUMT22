import math
import numpy as np
from data_representation import fp_transform, fps_transform
from distances import euclidean, city_block, sorensen, neyman


class KnnClassifier:
    def __init__(self, fps_train, loc_train, fps_test, loc_test):
        self.fps_train = fps_train
        self.loc_train = loc_train
        self.fps_test = fps_test
        self.loc_test = loc_test

        self.n_train = fps_train.shape[0]
        self.n_test = fps_test.shape[0]
        self.n_aps = fps_train.shape[1]
        self.min_dataset = np.min(fps_train)
        self.not_detected = 100

        self.fps_train_positive = fps_transform(fps_train, "positive", self.min_dataset, self.not_detected)
        self.fps_train_normalized = fps_transform(fps_train, "normalized", self.min_dataset, self.not_detected)
        self.fps_train_exponential = fps_transform(fps_train, "exponential", self.min_dataset, self.not_detected)
        self.fps_train_powed = fps_transform(fps_train, "powed", self.min_dataset, self.not_detected)

        self.fps_test_positive = fps_transform(fps_test, "positive", self.min_dataset, self.not_detected)
        self.fps_test_normalized = fps_transform(fps_test, "normalized", self.min_dataset, self.not_detected)
        self.fps_test_exponential = fps_transform(fps_test, "exponential", self.min_dataset, self.not_detected)
        self.fps_test_powed = fps_transform(fps_test, "powed", self.min_dataset, self.not_detected)

    def get_loc_test_samples(self, parameters):
        parameter_k = parameters[0]
        parameter_distance_function = parameters[1]
        parameter_data_representation = parameters[2]
        parameter_weights = parameters[3]

        l_estimated = np.zeros((self.n_test, 2))
        distances = self.estimate_fps_distances(parameter_distance_function, parameter_data_representation)

        for i in range(self.n_test):
            l_idx_neighbours, l_weights = self.get_neighbours(distances[i, :], parameter_k, parameter_weights)
            l_locs = []
            for idx in l_idx_neighbours:
                l_locs.append(self.loc_train[idx])
            l_estimated[i, :] = np.array(self.get_centroid(l_locs, l_weights))
        return l_estimated

    # Estimate the distance of all test samples to all training ones
    def estimate_fps_distances(self, dist_function, data_representation):
        d = np.zeros((self.n_test, self.n_train))
        for i in range(len(self.fps_test)):
            tst = self.get_test_fp(i, data_representation)
            for j in range(len(self.fps_train)):
                trn = self.get_train_fp(j, data_representation)
                if dist_function == "euclidean":
                    d[i, j] = euclidean(tst, trn)
                elif dist_function == "city_block":
                    d[i, j] = city_block(tst, trn)
                elif dist_function == "sorensen":
                    d[i, j] = sorensen(tst, trn)
                elif dist_function == "neyman":
                    d[i, j] = neyman(tst, trn)
                else:
                    print("ERROR!! Distance function " + dist_function + " unkwonn")
                    d[i, j] = math.inf
        return d

    def get_test_fp(self, ith, data_representation):
        tst = None
        if data_representation == "positive":
            tst = self.fps_test_positive[ith]
        elif data_representation == "normalized":
            tst = self.fps_test_normalized[ith]
        elif data_representation == "exponential":
            tst = self.fps_test_exponential[ith]
        elif data_representation == "powed":
            tst = self.fps_test_powed[ith]
        return tst

    def get_train_fp(self, ith, data_representation):
        trn = None
        if data_representation == "positive":
            trn = self.fps_train_positive[ith]
        elif data_representation == "normalized":
            trn = self.fps_train_normalized[ith]
        elif data_representation == "exponential":
            trn = self.fps_train_exponential[ith]
        elif data_representation == "powed":
            trn = self.fps_train_powed[ith]
        return trn

    # return nearest k neighbours and its weights
    def get_neighbours(self, d, k, weights):
        l_weights = []
        l_idx_neighbours = []
        for i in range(k):
            min_th = np.argmin(d)
            l_idx_neighbours.append(min_th)
            if d[min_th] == 0.0:
                d[min_th] = 0.0000001
            l_weights.append(1.0 / (d[min_th] ** weights))
            d[min_th] = math.inf
        return l_idx_neighbours, l_weights

    def get_centroid(self, l_locs, l_weights):
        centroid = [0, 0]
        i = 0
        for loc in l_locs:
            centroid[0] += loc[0] * l_weights[i]
            centroid[1] += loc[1] * l_weights[i]
            i += 1

        total_weight = 0
        for w in l_weights:
            total_weight += w
        centroid[0] = centroid[0] / total_weight
        centroid[1] = centroid[1] / total_weight

        return centroid

    def estimate_error(self, loc_estimated, loc_real):
        return euclidean(loc_estimated, loc_real)

    def estimate_mean_error(self, locs_estimated, locs_real):
        error = 0.0
        for i in range(len(locs_estimated)):
            error += euclidean(locs_estimated[i], locs_real[i])
        return error / len(locs_estimated)

