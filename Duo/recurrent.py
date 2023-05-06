# coding: utf-8
import os
import numpy as np
import csv
import fuzzer.new_fuzzer as nf

path = './result/outputs/'
target = 'tensorflow_caffe_pool1_0_2'
path += target
split_results = target.split('_', 2)
framwork_1 = split_results[0]
framwork_2 = split_results[1]
interface, _, mode = split_results[2].split('_', 2)

print(os.path.curdir)
if os.path.exists(path):

    ls = os.listdir(path)

    # 分离crash文件夹和数据文件
    crash_path = path + ls[0]
    ls = ls[1:]

    # 按照测试对象分离数据文件
    input_files_dict = {}
    output_files_dict = {}

    # 分离文件
    for file in ls:
        split_file = file.split('_', 3)
        file_name = split_file[0] + '_' + split_file[1] + '_' + split_file[2]
        slice_index = split_file[3]
        if split_file[0] == 'input':
            input_files_dict[file_name] = input_files_dict.get(file_name, [])
            input_files_dict[file_name].append(slice_index)
        elif split_file[0] == 'output':
            output_files_dict[file_name] = output_files_dict.get(file_name, [])
            output_files_dict[file_name].append(slice_index)

    # 将文件按照数值顺序排序，替代字典序
    for index, key in enumerate(input_files_dict):
        files = input_files_dict[key]
        files.sort(key=lambda x: int(x.split('.')[0]))

    for index, key in enumerate(output_files_dict):
        files = output_files_dict[key]
        files.sort(key=lambda x: int(x.split('.')[0]))

    # 组合还原
    input = {}
    output = {}

    for index, key in enumerate(output_files_dict):
        input[key] = []
        for file_slice in output_files_dict[key]:
            slice_path = path + '/' + key + '_' + file_slice
            with open(slice_path, encoding='utf-8') as f:
                data = np.loadtxt(f, float, delimiter=",")
            input[key].append(data)

    for index, key in enumerate(output_files_dict):
        output[key] = []
        for file_slice in output_files_dict[key]:
            slice_path = path + '/' + key + '_' + file_slice
            with open(slice_path, encoding='utf-8') as f:
                data = np.loadtxt(f, float, delimiter=",")
            output[key].append(data)

    # 请在此完成输入输出数据的格式调整，如果有必要（似乎上面直接读取的结果是float64，而我们的实验过程是float32
    input = np.array(input)
    output = np.array(output)
    print(input)
    print("***************************************************")
    print(output)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

    # 请在此调用已写好的test_one_input中的接口，复现故障数据
    nf.Fuzzer.test_one_input(1, input)
