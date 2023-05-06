import csv
import os
import time

import numpy as np


class DataWriter(object):
    def __init__(self, file_name, data):
        self.file_name = file_name
        self.data = data

    def write(self):
        for index in range(self.data.shape[0]):
            csv_file = open(self.file_name + '_' + str(index) + '.csv', 'w', newline='')
            write = csv.writer(csv_file)
            write.writerows(self.data[index])
            csv_file.close()


class CrashWriter(object):
    def __init__(self, file_path='./results'):
        self.path = file_path + '/crashes'
        self.file_name = self.path + '/crashes.csv'

        if os.path.exists(self.path):
            ls = os.listdir(self.path)
            for element in ls:
                remove_file = os.path.join(self.path, element)
                if not os.path.isdir(remove_file):
                    # print(remove_file)
                    os.remove(remove_file)
        else:
            os.makedirs(self.path)

        csv_file = open(self.file_name, 'w', newline='')
        write = csv.writer(csv_file)
        write.writerow(['ID', 'crash_style', 'max', 'mean', 'norm', 'record_time', 'baseID'])
        csv_file.close()

    def write_to_csv(self, id, crash_style, max, mean, norm, record_time, baseID):
        item = [id, crash_style, max, mean, norm, record_time, baseID]
        csv_file = open(self.file_name, 'a+', newline='')
        write = csv.writer(csv_file)
        write.writerow(item)
        csv_file.close()


class CSVResultWriter(object):
    def __init__(self, file_path, interface_name='', coverage_name='', sample_name='', power_schedule_name='',
                 generate_strategy=None, precision=0):
        self.path = file_path
        self.file_name = self.path + '/' + interface_name + '.csv'
        self.interface_name = interface_name
        self.coverage_name = coverage_name
        self.sample_name = sample_name
        self.power_schedule_name = power_schedule_name
        self.generate_strategy = generate_strategy
        self.precision = precision

        if os.path.exists(self.path):
            ls = os.listdir(self.path)
            for element in ls:
                remove_file = os.path.join(self.path, element)
                if not os.path.isdir(remove_file):
                    # print(remove_file)
                    os.remove(remove_file)
        else:
            os.makedirs(self.path)

        csv_file = open(self.file_name, 'w', newline='')
        write = csv.writer(csv_file)
        write.writerow(['ID', 'interface_name', 'coverage_name', 'sample_name', 'power_schedule_name', 'generate_strategy',
                        'precision', 'input_max', 'difference_max', 'difference_mean', 'difference_norm',
                        'record_time'])
        csv_file.close()

    def write_to_csv(self, id, input_max, data_max, data_mean, data_norm, data_time):
        item = [id, self.interface_name, self.coverage_name, self.sample_name, self.power_schedule_name, self.generate_strategy,
                self.precision, input_max, data_max, data_mean, data_norm, data_time]
        csv_file = open(self.file_name, 'a+', newline='')
        write = csv.writer(csv_file)
        write.writerow(item)
        csv_file.close()

    def write_statistical_results(self, num_execs, sum_elements, success_num, crashes, begin_time, end_time):
        item = [num_execs, sum_elements, success_num, crashes, begin_time, end_time]
        csv_file = open(self.file_name, 'a+', newline='')
        write = csv.writer(csv_file)
        write.writerow(['num_execs', 'sum_elements', 'success_num', 'crashes', 'begin_time', 'end_time'])
        write.writerow(item)
        csv_file.close()

    def getPath(self):
        return self.path


def WriteResults(mode, path, data, crash_flag, csvwriter, crashes, id, framework1_name, framework2_name,
                 input_difference=None, output_difference=None,
                 input_framework1=None, input_framework1_cpu=None, output_framework1=None, output_framework1_cpu=None,
                 input_framework2=None, input_framework2_cpu=None, output_framework2=None, output_framework2_cpu=None):
    # д������
    # tempwriter = DataWriter(path + '/data' + str(id), data)
    # tempwriter.write()

    # ��¼DLFramework1�ӿڵ��������
    # if input_framework1 is not None:
    #     tempwriter = DataWriter(path + '/input_' + framework1_name + str(id), input_framework1)
    #     tempwriter.write()
    # if input_framework1_cpu is not None:
    #     tempwriter = DataWriter(path + '/input_' + framework1_name + '_cpu' + str(id), input_framework1_cpu)
    #     tempwriter.write()
    # if output_framework1 is not None:
    #     tempwriter = DataWriter(path + '/output_' + framework1_name + str(id), output_framework1)
    #     tempwriter.write()
    # if output_framework1_cpu is not None:
    #     tempwriter = DataWriter(path + '/output_' + framework1_name + '_cpu' + str(id), output_framework1_cpu)
    #     tempwriter.write()
    #
    # # ��¼DLFramework2�ӿڵ��������
    # if input_framework2 is not None:
    #     tempwriter = DataWriter(path + '/input_' + framework2_name + str(id), input_framework2)
    #     tempwriter.write()
    # if input_framework2_cpu is not None:
    #     tempwriter = DataWriter(path + '/input_' + framework2_name + '_cpu' + str(id), input_framework2_cpu)
    #     tempwriter.write()
    # if output_framework2 is not None:
    #     tempwriter = DataWriter(path + '/output_' + framework2_name + str(id), output_framework2)
    #     tempwriter.write()
    # if output_framework2_cpu is not None:
    #     tempwriter = DataWriter(path + '/output_' + framework2_name + '_cpu' + str(id), output_framework2_cpu)
    #     tempwriter.write()
    # if crash_flag:
    #     # д��crashes����
    #     tempwriter = DataWriter(csvwriter.getPath() + '/crashes/data' + str(crashes), data)
    #     tempwriter.write()

    # д������Ԫ������
    WriteStatistic(id, input_difference, output_difference, csvwriter)


def WriteStatistic(id, input_difference, output_difference, csvwriter):
    data_id = id
    input_max = np.max(input_difference)
    data_max = None
    data_mean = None
    data_norm = None
    data_time = None
    if output_difference is not None:
        data_max = np.max(output_difference)
        data_mean = np.mean(output_difference)
        data_norm = np.linalg.norm(output_difference) / np.size(output_difference),
        data_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    csvwriter.write_to_csv(data_id, input_max, data_max, data_mean, data_norm, data_time)
