import numpy as np

POWER_BETA = 10
MAX_FACTOR = 5


def baisc_score(corpus, element):
    """
    :param corpus: the data got from caffe_fuzzer
    :param element:  the chosen one of corpus

    avg_exec_us:/exec_us:           the speed for this path
    bitmap_size:avg_bitmap_size:    the size of testcase
    handicap:                       how late in the game we learned about this path
    depth:                          input depth

    :return: Baisc perf score
    """

    perf_score = 40
    exec_us = element.speed
    avg_exec_us = np.average([item.speed for item in corpus])
    bitmap_size = element.data.size
    avg_bitmap_size = np.average([item.data.size for item in corpus])
    handicap = element.find_time

    if (exec_us * 0.1 > avg_exec_us):
        perf_score = 5
    elif (exec_us * 0.25 > avg_exec_us):
        perf_score = 10
    elif (exec_us * 0.5 > avg_exec_us):
        perf_score = 15
    elif (exec_us * 0.75 > avg_exec_us):
        perf_score = 30
    elif (exec_us * 4 < avg_exec_us):
        perf_score = 75
    elif (exec_us * 2 < avg_exec_us):
        perf_score = 50

    if (bitmap_size * 0.3 > avg_bitmap_size):
        perf_score *= 3
    elif (bitmap_size * 0.5 > avg_bitmap_size):
        perf_score *= 2
    elif (bitmap_size * 0.75 > avg_bitmap_size):
        perf_score *= 1.5
    elif (bitmap_size * 3 < avg_bitmap_size):
        perf_score *= 0.25
    elif (bitmap_size * 2 < avg_bitmap_size):
        perf_score *= 0.5
    elif (bitmap_size * 1.5 < avg_bitmap_size):
        perf_score *= 0.75

    if (handicap >= 4):
        perf_score *= 4
    elif (handicap):
        perf_score *= 2
    return perf_score


def next_p2(val):
    ret = 1
    while True:
        if ret < val:
            ret <<= 1
        else:
            return ret


def EXPLORE(corpus, element, power_beta=POWER_BETA, max_factor=MAX_FACTOR):
    perf_score = baisc_score(corpus, element)
    factor = 1
    perf_score *= factor / power_beta
    return perf_score


def EXPLOIT(corpus, element, power_beta=POWER_BETA, max_factor=MAX_FACTOR):
    perf_score = baisc_score(corpus, element)
    factor = max_factor
    perf_score *= factor / power_beta
    return perf_score


def COE(corpus, element, power_beta=POWER_BETA, max_factor=MAX_FACTOR):
    """
    :param n_fuzz:          Number of fuzz, does not overflow
    :param fuzz_total:      Total of n_fuzz
    :param n_paths:         Length of corpus
    :param fuzz_level:      Number of fuzzing iterations

    :return perf_score
    """
    n_fuzz = element.count
    fuzz_total = np.sum(item.count for item in corpus)
    n_paths = len(corpus)

    fuzz_level = n_fuzz
    perf_score = baisc_score(corpus, element)
    fuzz_mu = fuzz_total / n_paths
    if (n_fuzz <= fuzz_mu):
        if (fuzz_level < 16):
            factor = int((1 << fuzz_level))
        else:
            factor = max_factor
    else:
        factor = 0

    if factor > max_factor:
        factor = max_factor
    perf_score *= factor / power_beta
    return perf_score


def FAST(corpus, element, power_beta=POWER_BETA, max_factor=MAX_FACTOR):
    n_fuzz = element.count
    fuzz_level = n_fuzz
    perf_score = baisc_score(corpus, element)
    if fuzz_level < 16:
        if n_fuzz == 0:
            factor = int((1 << fuzz_level))
        else:
            factor = int((1 << fuzz_level)) / n_fuzz
    else:
        if n_fuzz == 0:
            factor = max_factor
        else:
            factor = max_factor / (next_p2(n_fuzz))

    if factor > max_factor:
        factor = max_factor
    perf_score *= factor / power_beta
    return perf_score


def LEANER(corpus, element, power_beta=POWER_BETA, max_factor=MAX_FACTOR):
    perf_score = baisc_score(corpus, element)
    n_fuzz = element.count + 1
    fuzz_level = n_fuzz
    if n_fuzz == 0:
        factor = fuzz_level
    else:
        factor = fuzz_level / n_fuzz

    if factor > max_factor:
        factor = max_factor
    perf_score *= factor / power_beta
    return perf_score


def QUAD(corpus, element, power_beta=POWER_BETA, max_factor=MAX_FACTOR):
    perf_score = baisc_score(corpus, element)
    n_fuzz = element.count
    fuzz_level = n_fuzz + 1
    if n_fuzz == 0:
        factor = fuzz_level * fuzz_level
    else:
        factor = fuzz_level * fuzz_level / n_fuzz

    if factor > max_factor:
        factor = max_factor
    perf_score *= factor / power_beta
    return perf_score
