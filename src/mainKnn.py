# ----------------------------------------
# MAIN
# ----------------------------------------
from KnnClassifier import KnnClassifier
from dataset import read_dataset
import time
import pandas as pd
import numpy as np


def do_exp(fps_train, loc_train, fps_test, loc_test):
    cl = KnnClassifier(fps_train, loc_train, fps_test, loc_test)

    l_parameters = []
    for k in range(1, 20, 2):
        for d in ["euclidean", "city_block", "sorensen", "neyman"]:
            for r in ["positive", "normalized", "exponential", "powed"]:
                for w in [0, 1, 2]:
                    l_parameters.append([k, d, r, w])

    l_res = []
    for parameters in l_parameters:
        t0 = time.process_time()
        loc_est = cl.get_loc_test_samples(parameters)
        mean_error = cl.estimate_mean_error(loc_est, loc_test)
        t = time.process_time() - t0
        l_res.append([parameters[0], parameters[1], parameters[2], parameters[3], mean_error, t])
        print([parameters[0], parameters[1], parameters[2], parameters[3], mean_error, t])

    return l_res


def do_dataset(dataset):
    fps_train, loc_train, fps_test, loc_test = read_dataset(dataset)

    print("Dataset: " + dataset)
    print("Train samples: " + str(fps_train.shape[0]) + " Test samples: " + str(fps_test.shape[0]))

    t0 = time.process_time()
    l_res = do_exp(fps_train, loc_train, fps_test, loc_test)
    print("Total Time: " + str(time.process_time() - t0))

    ith = np.argmin(np.array(l_res)[:, 4])
    print("Best: " + str(np.array(l_res)[ith, :]))

    out_filename = "../out/" + dataset + "_res.csv"
    df = pd.DataFrame(l_res)
    df.to_csv(out_filename, index=False, header=False)


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




