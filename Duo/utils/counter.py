import numpy as np


class CounterElement(object):
    def __init__(self, function_name, total, success):
        self.function_name = function_name
        self.total = total
        self.success = success
        if total <= 0:
            self.rate = 0
        else:
            self.rate = success / total

    def update(self, total_add, success_add):
        self.total += total_add
        self.success += success_add
        self.rate = self.total / self.rate


class InitCounter(object):
    def __init__(self, mutate_success_counter):
        self.mutate_success_counter = mutate_success_counter
        self.total_calling = 0
        for element in mutate_success_counter:
            self.total_calling += element[2]


class Counter(object):
    def __init__(self, initcounter):
        self.total_calling = initcounter.total_calling
        self.elements = []
        for index in range(len(initcounter.mutate_success_counter)):
            self.elements.append(CounterElement(initcounter.mutate_success_counter[index][0],
                                                initcounter.mutate_success_counter[index][1],
                                                initcounter.mutate_success_counter[index][2]))
        self.function_list = []

    def Search(self, function_name):
        """
        辅助函数，用于获取函数名在列表中的对应下标
        :param function_name: 函数名
        :return: 下标
        """
        for index in range(len(self.elements)):
            if function_name == self.elements[index].function_name:
                return index
        return None

    def get_first_funciton(self):
        """
        获取列表中成功率最高的扰动方法作为初始扰动方法。
        获取初始扰动方法的同时完成成功率排序，后续的排序过程在Update中完成
        :return: 扰动方法名
        """
        self.elements.sort(key=lambda element: element.rate, reverse=True)
        self.function_list.append(self.elements[0].function_name)
        return self.elements[0].function_name

    def get_next_function(self, p=0.1):
        """
        :param p: 预设的概率阈值
        获取任务链中的下一个扰动方法。
        :return: 扰动方法名
        """
        assert self.function_list
        cur_index = self.Search(self.function_list[-1])
        while 1:
            next_index = np.random.randint(0, len(self.elements))
            u = self.elements[next_index].rate
            if u >= min(pow(1 - p, abs(next_index - cur_index)), 1):
                self.function_list.append(self.elements[next_index].function_name)
                return self.elements[next_index].function_name

    def clear_functions(self):
        self.function_list.clear()

    def update(self, total_num, success_num):
        update_index = self.Search(self.function_list[-1])
        self.elements[update_index].update(total_add=total_num, success_add=success_num)
        self.elements.sort(key=lambda element: element.rate, reverse=True)
