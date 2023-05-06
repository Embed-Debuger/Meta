import os
import time

os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from collections import defaultdict
from utils.mutate_functions_by_bytes import do_mutate, get_list, mutate_precision
from utils.dectection_functions import check_difference, check_difference_framework, detection_1, detection_2, \
    detection_3, detection_nan
from utils.corpus import CorpusElement
from utils.counter import Counter
from utils.csv_writer import CrashWriter, WriteResults
from utils.power_schedules import *
from utils.monitors import Monitor
from utils.cafffe_compute import caffe_compute_all
from utils.pytorch_compute import pytorch_compute_all
from utils.tensorflow_compute import tensorflow_compute_all


class Fuzzer:

    def __init__(self, DLFramework, DLFramework_Other, input_corpus, coverage_function, dectection_function,
                 initcounter, powerschedule, precision, target_interface, csvwriter, mcmcflag=0, GPU_mode=0):
        """
        :param target_interface: 待测接口名
        :param DLFramework: 原深度学习框架
        :param DLFramework_Other: 比较深度学习框架
        :param input_corpus: 语料集
        :param initCounter: 初始化计数器
        :param powerschedule: 能量函数
        :param coverage_function: 覆盖方法
        :param dectection_function: 差分分析方法
        :param precision: 截断精度
        :param csvwriter: csv写入对象，将测试用例相关数据记录在本地
        :param mcmcflag: 是否采用mcmc策略
        :param GPU_mode: GPU模式,0--只是用GPU，1--采用GPU和CPU对比
        """
        self.DLFramework = DLFramework
        self.DLFramework_Other = DLFramework_Other
        self.DLFramework_Computer = self.decide_DLcomputer(self.DLFramework)
        self.DLFramework_Other_Computer = self.decide_DLcomputer(self.DLFramework_Other)
        # self.corpus_dir = corpus_dir 该参数融合到input_corpus中
        # self.sample_function = sample_function 该参数融合到input_corpus中
        self.coverage_funcntion = coverage_function
        self.edges = defaultdict(set)
        self.input_corpus = input_corpus
        # print('初始化完成, 当前corpus数量:', len(self.input_corpus.corpus))
        self.counter = Counter(initcounter)  # 计数器，主要用于MCMC过程
        self.power_schedule = powerschedule
        self.mcmcflag = mcmcflag
        self.dectection_function = dectection_function
        self.precision = precision
        self.target_interface = target_interface
        self.csvwriter = csvwriter
        self.crashwriter = CrashWriter(self.csvwriter.getPath())
        self.gpu_mode = GPU_mode
        # 额外计数器
        self.crashes = 0  # 崩溃次数
        self.monitor = Monitor(self.csvwriter.getPath())

    #  判断深度学习框架
    def decide_DLcomputer(self, DLFKname):
        if DLFKname == 'caffe':
            return caffe_compute_all
        elif DLFKname == 'tensorflow':
            return tensorflow_compute_all
        elif DLFKname == 'pytorch':
            return pytorch_compute_all
        return None

    #  生成新扰动元素
    def generate_inputs(self, sample_elements, mcmc_function=None):
        """
        修改后的生成函数，负责整个新元素的生成过程，包括是否采用mcmc进行扰动方法选择，能量函数分配和扰动方法执行过程
        :param sample_elements: 抽取出的语料元素
        :param mcmc_function: 仅用于采用mcmc策略的情况，mcmc策略选择出的扰动方法
        :return: 新生成的语料元素
        """
        # 如果无符合的抽取元素，不进行新元素生成
        if not sample_elements:
            return

        # 保存新生成的扰动元素
        new_corpus_elements = []

        # 不采用mcmc策略进行扰动方法选择，直接对抽取元素进行随机次变异，每次采用随机扰动方法
        if self.mcmcflag == 0:
            mutation_function_nums = np.random.randint(1, 6)  # 确定采用的扰动方法个数
            mutation_list = get_list()  # 获取全部扰动方法列表
            mutation_functions = []  # 记录选取的具体扰动方法

            # 测试代码，用于测试具体的扰动函数
            # mutation_functions.append(mutate_change_byte)

            # 实际操作代码
            for _ in range(mutation_function_nums):
                index = np.random.randint(0, len(mutation_list))  # 后期可改为function_selection函数
                mutation_functions.append(mutation_list[index])  # 添加扰动方法记录

            # 优先判断是否有能量函数
            if not self.power_schedule:
                for element in sample_elements:
                    start_time = time.clock()
                    mutated_data = self.mutate(element.data, mutation_functions)
                    end_time = time.clock()
                    # CorpusElement __init__(self, data, output,  coverage, parent, count=0, find_time=0, speed=0):
                    new_corpus_element = CorpusElement(data=mutated_data, output=None, coverage=None, parent=element,
                                                       count=0, find_time=end_time, speed=end_time - start_time)
                    new_corpus_elements.append(new_corpus_element)
            # O(n log(n))
            else:
                for element in sample_elements:
                    power = self.power_schedule(self.input_corpus.corpus, element)
                    print('power: ', power)
                    # power指定每个sample生成多少个新元素，故直接在此处进行循环
                    for _ in range(int(power)):
                        start_time = time.clock()
                        mutated_data = self.mutate(element.data, mutation_functions)
                        end_time = time.clock()
                        # CorpusElement __init__(self, data, output,  coverage, parent, count=0, find_time=0, speed=0):
                        new_corpus_element = CorpusElement(data=mutated_data, output=None, coverage=None,
                                                           parent=element,
                                                           count=0, find_time=end_time, speed=end_time - start_time)
                        new_corpus_elements.append(new_corpus_element)

        # 采用mcmc策略，每次进采用一个扰动方法
        else:
            if not self.power_schedule:  # 每个扰动方法执行一次
                for element in sample_elements:
                    start_time = time.clock()
                    mutated_data = self.mutate(element.data, [mcmc_function])
                    end_time = time.clock()
                    new_corpus_element = CorpusElement(data=mutated_data, output=None, coverage=None,
                                                       parent=element,
                                                       count=0, find_time=end_time, speed=end_time - start_time)
                    new_corpus_elements.append(new_corpus_element)

            else:
                for element in sample_elements:
                    power = self.power_schedule(self.input_corpus.corpus, element)
                    power = int(power)
                    print('power: ', power)
                    # power指定每个sample生成多少个新元素，故直接在此处进行循环
                    for _ in range(power):
                        start_time = time.clock()
                        mutated_data = self.mutate(element.data, [mcmc_function])
                        end_time = time.clock()
                        # CorpusElement __init__(self, data, output,  coverage, parent, count=0, find_time=0, speed=0):
                        new_corpus_element = CorpusElement(data=mutated_data, output=None, coverage=None,
                                                           parent=element,
                                                           count=0, find_time=end_time, speed=end_time - start_time)
                        new_corpus_elements.append(new_corpus_element)

        return new_corpus_elements

    # 变异过程入口，返回变异后的数据
    def mutate(self, data, function_list):
        """
        该函数是变异过程的入口函数
        :param data: 待变异数据
        :param function_list: 选取的变异方法列表
        :return: 变异后数据
        """
        if isinstance(data, np.ndarray):
            data = data.tolist()
        for mutation_function in function_list:
            do_mutate(data, mutation_function)
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        return data

    # 新样例分析
    def test_one_input(self, id, corpus_element):
        """
        判断是否为新覆盖
        :param corpus_element:
        :return:
        """
        data = corpus_element.data
        data = mutate_precision(data, precision=self.precision)
        if np.isnan(data.all()):
            return False
        # 计算输出结果
        input_framework1, output_framework1, input_framework1_cpu, output_framework1_cpu = None, None, None, None
        outputs_framework1, inputs_framework1, outputs_framework1_cpu, inputs_framework1_cpu = \
            self.DLFramework_Computer(data, self.target_interface, self.gpu_mode, self.DLFramework_Other)
        if self.gpu_mode != 0:
            input_framework1_cpu, output_framework1_cpu = inputs_framework1_cpu[0], outputs_framework1_cpu[0]
        if self.gpu_mode != 2:
            input_framework1, output_framework1 = inputs_framework1[0], outputs_framework1[0]

        input_framework2, output_framework2, input_framework2_cpu, output_framework2_cpu = None, None, None, None
        outputs_framework2, inputs_framework2, outputs_framework2_cpu, inputs_framework2_cpu = \
            self.DLFramework_Other_Computer(data, self.target_interface, self.gpu_mode, self.DLFramework)
        if self.gpu_mode != 0:
            input_framework2_cpu, output_framework2_cpu = inputs_framework2_cpu[0], outputs_framework2_cpu[0]
        if self.gpu_mode != 2:
            input_framework2, output_framework2 = inputs_framework2[0], outputs_framework2[0]
        # check input on GPU/CPU
        if self.gpu_mode == 1 and (
                not check_difference_framework(input_framework1, input_framework1_cpu, self.precision)):
            return False
        # 计算覆盖
        if self.coverage_funcntion:
            if self.gpu_mode == 2:
                coverage = self.coverage_funcntion(self.input_corpus.corpus, output_framework1_cpu,
                                                   len(output_framework1_cpu.shape))
            else:
                coverage = self.coverage_funcntion(self.input_corpus.corpus, output_framework1,
                                                   len(output_framework1.shape))
        else:
            coverage = None
        # 采用差分测试
        if self.dectection_function is not None:
            # 进行输入差分分析
            input_difference = None
            if self.gpu_mode == 2:
                input_difference = self.dectection_function(input_framework1_cpu, input_framework2_cpu)
            else:
                input_difference = self.dectection_function(input_framework1, input_framework2)
            # check Framework
            if not check_difference(input_difference, self.precision):
                return False
            # check mode
            if self.gpu_mode == 1 and (
                    not check_difference_framework(input_framework2, input_framework2_cpu, self.precision)):
                return False

            # 进行接口输出差分测试
            output_difference = None
            # 根据差分结果判断是否满足目标
            crash_flag = 0
            if self.gpu_mode == 0:
                output_difference = self.dectection_function(output_framework1, output_framework2)
                crash_flag = detection_3(self.crashwriter, self.precision, self.crashes, output_difference, baseID=id)
            elif self.gpu_mode == 1:
                output_framework1_mode_difference = output_framework1 - output_framework1_cpu
                output_framework2_mode_difference = output_framework2 - output_framework2_cpu
                output_difference_1 = self.dectection_function(output_framework1, output_framework2)
                output_difference_2 = self.dectection_function(output_framework1, output_framework2_cpu)
                output_difference_3 = self.dectection_function(output_framework1_cpu, output_framework2)
                output_difference_4 = self.dectection_function(output_framework1_cpu, output_framework2_cpu)
                crash_flag = detection_1(self.crashwriter, self.precision, self.crashes, self.DLFramework,
                                         self.DLFramework_Other, output_framework1_mode_difference,
                                         output_framework2_mode_difference, output_difference_1, output_difference_2,
                                         output_difference_3, output_difference_4, baseID=id)
            elif self.gpu_mode == 2:
                output_difference = self.dectection_function(output_framework1_cpu, output_framework2_cpu)
                crash_flag = detection_2(self.crashwriter, self.precision, self.crashes, output_difference, baseID=id)

            crash_flag |= detection_nan(self.crashwriter, self.crashes,
                                        input_framework1=input_framework1,
                                        input_framework1_cpu=input_framework1_cpu,
                                        input_framework2=input_framework2,
                                        input_framework2_cpu=input_framework2_cpu,
                                        output_framework1=output_framework1,
                                        output_framework1_cpu=output_framework1_cpu,
                                        output_framework2=output_framework2,
                                        output_framework2_cpu=output_framework2_cpu, baseID=id)
            if crash_flag:
                self.crashes += 1

            # 记录实验结果
            WriteResults(self.gpu_mode, self.csvwriter.getPath(), data, crash_flag, self.csvwriter, self.crashes,
                         id, self.DLFramework, self.DLFramework_Other,
                         input_framework1=input_framework1,
                         output_framework1=output_framework1,
                         input_framework2=input_framework2,
                         output_framework2=output_framework2,
                         input_framework2_cpu=input_framework2_cpu,
                         output_framework2_cpu=output_framework2_cpu,
                         input_framework1_cpu=input_framework1_cpu,
                         output_framework1_cpu=output_framework1_cpu,
                         input_difference=input_difference,
                         output_difference=output_difference)
        # 不采用差分测试
        else:
            crash_flag = detection_nan(self.crashwriter, self.crashes,
                                       input_framework1=input_framework1,
                                       input_framework1_cpu=input_framework1_cpu,
                                       input_framework2=input_framework2,
                                       input_framework2_cpu=input_framework2_cpu,
                                       output_framework1=output_framework1,
                                       output_framework1_cpu=output_framework1_cpu,
                                       output_framework2=output_framework2,
                                       output_framework2_cpu=output_framework2_cpu, baseID=id)
            if crash_flag:
                self.crashes += 1
                print('发现NaN错误')
        #  Adding
        corpus_element.coverage = coverage
        corpus_element.output = output_framework1

        # 判断是否为新覆盖
        has_new = self.input_corpus.maybe_add_to_corpus(corpus_element)
        return has_new

    def fuzz(self):
        """ 模糊测试执行过程 """
        # demo阶段采用定量模糊测试手段进行效率对比
        # 额外计数参数
        max_execs = 10  # 模糊测试总尝试次数
        num_execs = 0  # 当前执行次数
        success_num = 0  # 成功生成语料集元素
        sum_elements = 0  # 生成的总语料集元素个数
        # 采用mcmc
        begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if self.mcmcflag:
            self.counter.clear_functions()

            # 直接分离出第一步，减少代码中的条件判断
            sample_elements = self.input_corpus.sample_input()
            first_function = self.counter.get_first_funciton()
            new_elements = self.generate_inputs(sample_elements, mcmc_function=first_function)  # 将两种生成策略合并为一种生成策略
            sum_elements += len(new_elements)
            success_elements_num = 0  # success_elements_num为增量
            for element in new_elements:
                num_execs += 1
                has_new = self.test_one_input(num_execs, element)
                if has_new:
                    success_elements_num += 1
                    success_num += 1
                    # print("%s/%s-%s(%d): %d" % (
                    # self.DLFramework, self.DLFramework_Other, self.target_interface, self.gpu_mode, num_execs))
            self.counter.update(total_num=len(new_elements), success_num=success_elements_num)

            # 已选出第一个函数的条件下选取第二个函数
            while num_execs <= max_execs:
                assert self.input_corpus.corpus
                self.monitor.logging("start: " + str(num_execs))
                sample_elements = self.input_corpus.sample_input()
                self.monitor.logging("sample")
                next_function = self.counter.get_next_function()
                self.monitor.logging("generate_start")
                new_elements = self.generate_inputs(sample_elements, mcmc_function=next_function)  # 将两种生成策略合并为一种生成策略
                self.monitor.logging("generate_end")
                sum_elements += len(new_elements)
                success_elements_num = 0
                for element in new_elements:
                    num_execs += 1
                    self.monitor.logging("test_start " + str(num_execs))
                    has_new = self.test_one_input(num_execs, element)  # 待修改
                    self.monitor.logging("test_end " + str(num_execs))
                    if has_new:
                        success_elements_num += 1
                        success_num += 1
                        # print("%s/%s-%s(%d): %d" % (
                        # self.DLFramework, self.DLFramework_Other, self.target_interface, self.gpu_mode, num_execs))
                self.counter.update(total_num=len(new_elements), success_num=success_elements_num)

        # 不采用mcmc
        else:
            while num_execs <= max_execs:
                assert self.input_corpus.corpus
                sample_elements = self.input_corpus.sample_input()
                new_elements = self.generate_inputs(sample_elements)
                sum_elements += len(new_elements)
                for element in new_elements:
                    num_execs += 1
                    has_new = self.test_one_input(num_execs, element)
                    if has_new:
                        success_num += 1
                    # print("%s/%s-%s(%d): %d" % (
                    # self.DLFramework, self.DLFramework_Other, self.target_interface, self.gpu_mode, num_execs))
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("%s/%s-%s(%d)生成的总数据：%d" % (
            self.DLFramework, self.DLFramework_Other, self.target_interface, self.gpu_mode, sum_elements))
        print("%s/%s-%s(%d)有效的数据: %d" % (
            self.DLFramework, self.DLFramework_Other, self.target_interface, self.gpu_mode, success_num))
        print("%s/%s-%s(%d)发现的crash: %d" % (
            self.DLFramework, self.DLFramework_Other, self.target_interface, self.gpu_mode, self.crashes))
        self.csvwriter.write_statistical_results(num_execs, sum_elements, success_num, self.crashes, begin_time,
                                                 end_time)
