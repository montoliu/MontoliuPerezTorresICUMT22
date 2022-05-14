import math


def fp_transform(fp, method, min_dataset, not_detected, alpha=24, beta=math.e):
    if method == "positive":
        fp_out = [positive(x, min_dataset, not_detected) for x in fp]
    elif method == "normalized":
        fp_out = [normalized(x, min_dataset, not_detected) for x in fp]
    elif method == "exponential":
        fp_out = [exponential(x, min_dataset, not_detected, alpha) for x in fp]
    elif method == "powed":
        fp_out = [powed(x, min_dataset, not_detected, beta) for x in fp]
    return fp_out


def fps_transform(fps, method, min_dataset, not_detected, alpha=24, beta=math.e):
    l_fps = []
    for fp in fps:
        fp_out = fp_transform(fp, method, min_dataset, not_detected)
        l_fps.append(fp_out)
    return l_fps


def positive(rssi, min_dataset, not_detected):
    if rssi == not_detected:
        return 0
    return rssi - min_dataset


def normalized(rssi, min_dataset, not_detected):
    if rssi == not_detected:
        return 0
    return positive(rssi, min_dataset, not_detected) / -min_dataset


def exponential(rssi, min_dataset, not_detected, alpha):
    if rssi == not_detected:
        return 0
    x = positive(rssi, min_dataset, not_detected)
    x = math.exp(x/alpha) / math.exp(-min_dataset / alpha)
    return x


def powed(rssi, min_dataset, not_detected, beta):
    if rssi == not_detected:
        return 0
    x = positive(rssi, min_dataset, not_detected)
    x = (x**beta) / ((-min_dataset)**beta)
    return x




