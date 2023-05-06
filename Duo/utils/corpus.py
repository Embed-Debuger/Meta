import logging
import time

#import caffe
import pyflann
# import cv2
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np

from utils.coverage_functions import *

_BUFFER_SIZE = 50

tf.disable_v2_behavior()
class CorpusElement(object):
    '''
    单个测试用例类
    '''

    def __init__(self, data, output=None, coverage=None, parent=None, count=0, find_time=0, speed=0):
        self.data = data
        self.output = output
        self.coverage = coverage
        self.parent = parent
        self.count = count
        self.find_time = find_time
        self.speed = speed

    def oldest_ancestor(self):
        current_element = self
        generations = 0
        while current_element.parent is not None:
            current_element = current_element.parent
            generations += 1
        return current_element, generations


class InputCorpus(object):

    def __init__(self, seed_corpus, sample_function, coverage_function, threshold, algorithm):

        self.mutations_processed = 0
        self.corpus = []
        self.sample_function = sample_function
        self.coverage_function = coverage_function
        self.start_time = time.time()
        self.current_time = time.time()
        self.log_time = time.time()

        for corpus_element in seed_corpus:
            self.corpus.append(corpus_element)

        if coverage_function == neuron_coverage_function:
            coverage_function.bounds = getBounds(self.corpus)
            coverage = self.corpus[-1].coverage
            self.updater = NeuronUpdater(coverage)
        elif not coverage_function:
            self.updater = NoneUpdater()
        else:
            self.updater = Updater(threshold, algorithm)
            self.updater.build_index_and_flush(self)

    def maybe_add_to_corpus(self, element):
        self.mutations_processed += 1
        return self.updater.update_function(self, element)
        # if self.coverage_function == neuron_coverage_function:
        #     self.updater.update_function(self, element)
        # elif self.coverage_function == None:
        #     self.updater.update_function(self, element)
        # else:
        #     return self.updater.update_function(self, element)

        # current_time = time.time()

    def sample_input(self):
        return self.sample_function(self)


class NoneUpdater(object):
    def __init__(self):
        self.coverage = None

    def update_function(self, corpus_object, element):
        corpus_object.corpus.append(element)
        if len(corpus_object.corpus) > 500:
            corpus_object.corpus.remove(corpus_object.corpus[0])
        return True


class NeuronUpdater(object):
    def __init__(self, coverage):
        self.coverage = coverage

    def update_function(self, corpus_object, element):
        has_new = False
        if (element.coverage[0] > self.coverage[0]
                or element.coverage[1] > self.coverage[1]
                or element.coverage[2] > self.coverage[2]
                or element.coverage[3] > self.coverage[3]):
            corpus_object.corpus.append(element)
            has_new = True
            return has_new

        for index in range(len(corpus_object.corpus)):
            if corpus_object.corpus[index].coverage[4] != []:
                if corpus_object.corpus[index].coverage[4] == element.coverage[4]:
                    return has_new
        has_new = True
        corpus_object.corpus.append(element)
        return has_new


class Updater(object):

    def __init__(self, threshold, algorithm):
        self.flann = pyflann.FLANN()
        self.threshold = threshold
        self.algorithm = algorithm
        self.corpus_buffer = []
        self.lookup_array = []

    def build_index_and_flush(self, corpus_object):

        self.corpus_buffer[:] = []
        self.lookup_array = np.array(
            [element.coverage for element in corpus_object.corpus]
        )

        self.flann.build_index(self.lookup_array, algorithm=self.algorithm)

    def update_function(self, corpus_object, element):

        if corpus_object.corpus is None:
            corpus_object.corpus = [element]
            self.build_index_and_flush(corpus_object)
        else:

            _, approx_distance = self.flann.nn_index(
                np.array([element.coverage]), 1, algorithm=self.algorithm
            )

            exact_distances = [
                np.sum(np.square(element.coverage - buffer_elt))
                for buffer_elt in self.corpus_buffer
            ]
            # print("exct:", exact_distances)
            # if len(exact_distances) > 0 and np.isnan(exact_distances[0]):
            #     exact_distances[0] = 0
            # print("coverage:", element.coverage)
            # print("exact:", exact_distances)
            # print("appro:", approx_distance.tolist())
            nearest_distance = min(exact_distances + approx_distance.tolist())
            # print("neraest_distance:", nearest_distance)
            has_new = False
            # if nearest_distance > self.threshold or np.isnan(nearest_distance):
            if nearest_distance > self.threshold:
                corpus_object.corpus.append(element)
                self.corpus_buffer.append(element.coverage)
                if len(self.corpus_buffer) >= _BUFFER_SIZE:
                    self.build_index_and_flush(corpus_object)
                has_new = True
            return has_new


def seed_corpus(inputs, target, coverage_function):
    seed_corpus = []
    for input in inputs:
        if not coverage_function:
            output = None
            coverage = None
        else:
            # output = target(input)
            output = target(input)[0]
            shape_length = len(output.shape)
            # print(shape_length)
            # print(output.shape[0])
            # print(output)
            coverage = coverage_function(output, shape_length)
            # print(coverage)
        new_element = CorpusElement(data=input, output=output, coverage=coverage, parent=None)
        seed_corpus.append(new_element)
    return seed_corpus


def generate_seed_corpus(corpus_dir, target, coverage_function, target_interface, GPU_mode):
    to_import = list(corpus_dir.iterdir())
    if not to_import:
        logging.error('No corpus found')
        exit()
    inputs = []
    for path in to_import:
        # print(path)
        inputs.append(import_testcase(path))

    seed_corpus = []
    for input in inputs:
        # 此处仅考虑在CPU模式下的目标输出值即可
        # input = input.astype(np.float64)
        input = np.random.normal(np.random.randint(-100, 100, 1), np.random.randint(100, 10000, 1), 28 * 28 * 3).reshape(28, 28, 3)
        # output = target(input, target_interface, 2)[2]
        # shape_length = len(output.shape)
        # coverage = coverage_function([], output, shape_length)
        new_element = CorpusElement(data=input, parent=None)
        seed_corpus.append(new_element)
    return seed_corpus


def generate_seed_corpus_as_iterator(corpus_dir, index, target, coverage_function, target_interface, GPU_mode):
    to_import = list(corpus_dir.iterdir())
    if not to_import:
        logging.error('No corpus found')
        exit()
    inputs = []
    for path in to_import:
        inputs.append(import_testcase(path))

    seed_corpus = []
    # 随机获取一个对象生成corpus
    input = inputs[index if index < len(inputs) else index % len(inputs)]
    output = target(input, target_interface, GPU_mode)[0]
    shape_length = len(output.shape)
    coverage = coverage_function([], output, shape_length)
    new_element = CorpusElement(data=input, output=output, coverage=coverage, parent=None)
    seed_corpus.append(new_element)
    return seed_corpus

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
# config.gpu_options.allow_growth = True #allocate dynamically
# sess = tf.Session(config = config)

def import_testcase(path):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # testcase1 = caffe.io.load_image(str(path))
    # testcase1 = caffe.io.resize(testcase1, (28, 28, 3))
    with tf.Session() as sess:
        # tf.device('/gpu')
        sess.run(tf.global_variables_initializer())
        source = tf.read_file(str(path))
        testcase2 = tf.image.decode_jpeg(source, channels=3)
        testcase2 = tf.image.resize_images(testcase2, [28, 28])
        input_tensorflow = testcase2.eval()
    return input_tensorflow
