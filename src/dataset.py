import math
import random
import pandas as pd
from scipy.spatial.distance import euclidean


def path_loss(d, r0, gamma, d0, factor):
    epsilon = (random.uniform(0, 1)-0.5)*2.0*factor
    return r0 - 10 * gamma * math.log(d/d0, 10) + epsilon


def generate_samples(l_points, l_std, l_aps, samples_by_point, r0, gamma, d0):
    l_rss = []
    l_crd = []
    i = 0
    for p in l_points:
        for s in range(samples_by_point):
            rss = []
            for a in l_aps:
                d = euclidean(p, a)
                rss.append(int(path_loss(d, r0, gamma, d0, l_std[i])))
            l_crd.append([p[0], p[1], 0, 0, 0])
            l_rss.append(rss)
        i += 1
    return l_rss, l_crd


def generate_dataset(dataset_name, samples_by_point, r0, gamma, d0,
                     l_train_points, l_test_points, l_train_std, l_test_std, l_aps):
    l_train_rss, l_train_crd = generate_samples(l_train_points, l_train_std, l_aps, samples_by_point, r0, gamma, d0)
    l_test_rss, l_test_crd = generate_samples(l_test_points, l_test_std, l_aps, samples_by_point, r0, gamma, d0)

    filename_train_rss = "../data/" + dataset_name + "/" + dataset_name + "_trnrss.csv"
    filename_train_crd = "../data/" + dataset_name + "/" + dataset_name + "_trncrd.csv"
    filename_test_rss = "../data/" + dataset_name + "/" + dataset_name + "_tstrss.csv"
    filename_test_crd = "../data/" + dataset_name + "/" + dataset_name + "_tstcrd.csv"

    df = pd.DataFrame(l_train_rss)
    df.to_csv(filename_train_rss, index=False, header=False)

    df = pd.DataFrame(l_train_crd)
    df.to_csv(filename_train_crd, index=False, header=False)

    df = pd.DataFrame(l_test_rss)
    df.to_csv(filename_test_rss, index=False, header=False)

    df = pd.DataFrame(l_test_crd)
    df.to_csv(filename_test_crd, index=False, header=False)


def read_dataset(dataset_name):
    filename_train_rss = "../data/" + dataset_name + "/" + dataset_name + "_trnrss.csv"
    filename_train_crd = "../data/" + dataset_name + "/" + dataset_name + "_trncrd.csv"
    filename_test_rss = "../data/" + dataset_name + "/" + dataset_name + "_tstrss.csv"
    filename_test_crd = "../data/" + dataset_name + "/" + dataset_name + "_tstcrd.csv"

    df = pd.read_csv(filename_train_rss, header=None)
    l_train_rss = df.values

    df = pd.read_csv(filename_train_crd, header=None)
    l_train_crd = df.values

    df = pd.read_csv(filename_test_rss, header=None)
    l_test_rss = df.values

    df = pd.read_csv(filename_test_crd, header=None)
    l_test_crd = df.values

    return l_train_rss, l_train_crd[:, 0:2], l_test_rss, l_test_crd[:, 0:2]
