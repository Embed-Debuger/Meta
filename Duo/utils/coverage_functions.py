import numpy as np

global bound
bound = []


def absolute_coverage_function(corpus, output, shape_length):
    '''
    基于输出值得绝对值之和计算覆盖
    :param coverages_batches:
    :return:
    '''
    coverage = 0
    if shape_length == 1:
        for idx in range(output.shape[0]):
            coverage += np.abs(output[idx])
    elif shape_length == 3:
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for k in range(output.shape[2]):
                    coverage += np.abs(output[i][j][k])

    return np.array([coverage])


def raw_coverage_function(inputs, output, shape_length):
    coverage = 0
    if shape_length == 1:
        for idx in range(output.shape[0]):
            coverage += output[idx]
    elif shape_length == 3:
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for k in range(output.shape[2]):
                    coverage += output[i][j][k]
    return np.array([coverage])


class Bounder:
    def __init__(self, lower=0, upper=0):
        self.upper = upper
        self.lower = lower


def getBounds(corpus):
    row = corpus[-1].data.shape[0]
    col = corpus[-1].data.shape[1]
    bound = []

    # for iter1 in range(row):
    #     b = []
    #     for iter2 in range(col):
    #         b.append(Bounder(0, 255))
    #     bound.append(b)

    max = np.zeros([row, col])
    min = np.zeros([row, col])

    for iter1 in range(row):
        for iter2 in range(col):
            min[iter1][iter2] = 255

    for output in corpus:
        for iter1 in range(row):
            for iter2 in range(col):
                element = output.data[iter1][iter2][0]
                if (element > max[iter1][iter2]):
                    max[iter1][iter2] = element
                if (element < min[iter1][iter2]):
                    min[iter1][iter2] = element

    for iter1 in range(row):
        b = []
        for iter2 in range(col):
            b.append(Bounder(min[iter1][iter2], max[iter1][iter2]))
        bound.append(b)
    return bound


def neuron_coverage_function(corpus, output, shape_length):
    global bound
    if not (bound):
        bound = getBounds(corpus)
    k = 100
    row = output.data.shape[1]
    col = output.data.shape[2]
    # KMN
    KMN_cnt = np.zeros([row, col, k])

    for element in corpus:
        for iter1 in range(row):
            for iter2 in range(col):
                min = bound[iter1][iter2].lower
                max = bound[iter1][iter2].upper
                len = max - min + 0.1
                if (element.data[iter1][iter2][0] >= min and element.data[iter1][iter2][0] <= max):
                    index = (element.data[iter1][iter2][0] - min) / (len / k)
                    index = int(index)
                    KMN_cnt[iter1][iter2][index] = 1
    cnt = 0
    for iter1 in range(row):
        for iter2 in range(col):
            min = bound[iter1][iter2].lower
            max = bound[iter1][iter2].upper
            len = max - min + 0.1
            if (output[0][iter1][iter2] >= min and output[0][iter1][iter2] <= max):
                index = (output[0][iter1][iter2] - min) / (len / k)
                index = int(index)
                KMN_cnt[iter1][iter2][index] = 1
            for iter3 in range(k):
                if (KMN_cnt[iter1][iter2][iter3]):
                    cnt = cnt + 1
    KMN = float(cnt) / (row * col * k)

    # NB/SNA
    NB_cnt = np.zeros([row, col, 2])
    SNA_cnt = np.zeros([row, col])

    for element in corpus:
        for iter1 in range(row):
            for iter2 in range(col):
                min = bound[iter1][iter2].lower
                max = bound[iter1][iter2].upper
                if element.data[iter1][iter2][0] < min:
                    NB_cnt[iter1][iter2][0] = 1
                if element.data[iter1][iter2][0] > max:
                    NB_cnt[iter1][iter2][1] = 1
                    SNA_cnt[iter1][iter2] = 1
    cnt_NB = 0
    cnt_SNA = 0
    for iter1 in range(row):
        for iter2 in range(col):
            min = bound[iter1][iter2].lower
            max = bound[iter1][iter2].upper
            if output[0][iter1][iter2] < min:
                NB_cnt[iter1][iter2][0] = 1
            if output[0][iter1][iter2] > max:
                NB_cnt[iter1][iter2][1] = 1
                SNA_cnt[iter1][iter2] = 1
            if NB_cnt[iter1][iter2][0]:
                cnt_NB = cnt_NB + 1
            if NB_cnt[iter1][iter2][1]:
                cnt_NB = cnt_NB + 1
            if SNA_cnt[iter1][iter2]:
                cnt_SNA = cnt_SNA + 1
    NB = cnt_NB / (row * col * 2)
    SNA = float(cnt_SNA) / (row * col)

    # TKN/TKNPat
    TKN_cnt = np.zeros([row, col])
    TKNPat = []
    for element in corpus:
        TKN_tuple = []
        for iter1 in range(row):
            for iter2 in range(col):
                TKN_tuple.append((element.data[iter1][iter2][0], iter1, iter2))
        TKN_tuple.sort(key=lambda num: num[0], reverse=True)
        for index in range(k):
            TKN_cnt[TKN_tuple[index][1]][TKN_tuple[index][2]] = 1

    cnt = 0
    TKN_tuple = []
    for iter1 in range(row):
        b = []
        for iter2 in range(col):
            b.append(0)
            TKN_tuple.append((output[0][iter1][iter2], iter1, iter2))
        TKNPat.append(b)
    TKN_tuple.sort(key=lambda num: num[1], reverse=True)

    for index in range(k):
        TKN_cnt[TKN_tuple[index][1]][TKN_tuple[index][2]] = 1
        TKNPat[TKN_tuple[index][1]][TKN_tuple[index][2]] = 1

    for iter1 in range(row):
        for iter2 in range(col):
            if (TKN_cnt[iter1][iter2]):
                cnt = cnt + 1
    TKN = float(cnt) / (row * col)

    cov_list = [KMN, NB, SNA, TKN, TKNPat]
    return cov_list


def neuron_coverage_origin_function(corpus, output, shape_length):
    cov_list = [0, 0, 0, 0, []]
    return cov_list
