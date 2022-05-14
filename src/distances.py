from scipy.spatial import distance


def euclidean(f1, f2):
    return distance.euclidean(f1, f2)


def city_block(f1, f2):
    return distance.cityblock(f1, f2)


def sorensen(f1, f2):
    num = 0
    den = 0
    for i in range(len(f1)):
        num += abs(f1[i] - f2[i])

    for i in range(len(f1)):
        den += f1[i] + f2[i]

    if den == 0:
        a= 1

    return num / den


def neyman(f1, f2):
    d = 0
    for i in range(len(f1)):
        if f1[i] != 0:
            d += ((f1[i] - f2[i])**2) / f1[i]
    return d

