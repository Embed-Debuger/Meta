#from utils.caffe_compute import *
from utils.dectection_functions import difference_general
from utils.pytorch_compute import *
from utils.tensorflow_compute import *


def decide_dectection_function(DLFramework, DLFramework_Other, interface):
    # if DLFramework == 'caffe':
    #     return difference_caffe_other
    # if DLFramework_Other == 'caffe':
    #     return difference_other_caffe
    if interface == 'dense1':
        return None
    return difference_general


def decide_seed_corpus_target(DLFramework):
    if DLFramework == 'caffe':
        return caffe_compute_single
    elif DLFramework == 'tensorflow':
        return tensorflow_compute_single
    elif DLFramework == 'pytorch':
        return pytorch_compute_single


def decide_computer(DLFramework):
    if DLFramework == 'caffe':
        return caffe_compute_single
    elif DLFramework == 'tensorflow':
        return tensorflow_compute_single
    elif DLFramework == 'pytorch':
        return pytorch_compute_single
