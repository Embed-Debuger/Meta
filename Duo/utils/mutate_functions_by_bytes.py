import string

import cv2
import numpy as np
import random
# from pomegranate import *
# from skimage import img_as_ubyte

from utils.data_conversions import *
from utils.process_after_mutation import *


# from utils import common


def do_mutate(data_list, mutate_function):
    """
    该函数为具体执行扰动方法的变异过程。
    :param data_list: 待变异数据
    :param mutate_function:选取的扰动方法
    :return: 变异后数据
    """

    data_list = mutate_function(data_list)
    return data_list


def get_list():
    # 设置为元组，保证其不可变
    # target_mutaion_functions = (mutate_erase_bytes,
    #                             mutate_insert_bytes,
    #                             mutate_change_byte,
    #                             mutate_insert_repeated_bytes,
    #                             mutate_change_ascii_integer,
    #                             mutate_change_bit,
    #                             mutate_erase_bit,
    #                             mutate_insert_bit
    #                             mutate_white_noise,
    #                             mutate_rotate,
    #                             mutate_scale,
    #                             mutate_triangular_matrix,
    #                             mutate_kernel_matrix,
    #                             mutate_nan,
    #                             mutate_inf,
    #                             mutate_high_number,
    #                             mutate_high_precision
    #                             )
    target_mutaion_functions = [mutate_scale]
    #target_mutaion_functions = (mutate_white_noise,mutate_change_byte)
    return_list = list(target_mutaion_functions)
    return return_list


def mutate_for_li(li, target_id):
    """
    :param li:
    :param target_id: 具体变异方法的编号
    :return:
    """
    mutation_list = get_list()
    return mutation_list[target_id](li)


def list_to_byte(data, prob1=0.3, prob2=0.3, prob3=0.3):
    """
    :param data: 待变异数据, list格式
    :param prob1: 变异概率1
    :param prob2: 变异概率2
    :param prob3: 变异概率3
    :return: 定位的具体二进制值
    """
    data = np.array(data)
    for i in range(data.shape[0]):
        if random.random() > prob1:
            continue
        for j in range(data.shape[1]):
            if random.random() > prob2:
                continue
            for k in range(data.shape[2]):
                if random.random() < prob3:
                    num = data[i][j][k]
                    num = converse(num, float_to_byte)
                    return num, [i, j, k]

    num = data[0][0][0]
    num = converse(num, float_to_byte)
    return num, [0, 0, 0]


def byte_to_list(data, num, location):
    # 处理
    i = location[0]
    j = location[1]
    k = location[2]
    num = process(num)
    # 转为double类型
    num = converse(num, byte_to_float)
    data[i][j][k] = num
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    data = np.clip(data, a_min=0.0, a_max=1.0)
    return data


def mutate_sift(data, constraint=None, threshold=0.00, a_min=-1.0, a_max=1.0, ):
    SIGMA_CONSTANT = 15

    data = mutate_white_noise(data, sigma=0.2)
    keypoints = getSiftKeypoints(data)
    data_size = []

    data_size.append(len(data))
    element = data[0]

    data_size.append(len(element))
    element = element[0]

    data_size.append(len(element))
    # 图像调整
    for _ in range(np.random.randint(low=1, high=3)):
        index = np.random.randint(low=0, high=len(keypoints))
        mu_x, mu_y, sigma = int(round(keypoints[index][0].pt[0])), int(round(keypoints[index][0].pt[1])), \
                            keypoints[index][2]
        sigma += SIGMA_CONSTANT
        # 对点做正态分布
        d_x = NormalDistribution(mu_x, sigma)
        d_y = NormalDistribution(mu_y, sigma)
        x = int(d_x.sample())
        y = int(d_y.sample())

        if x >= data_size[0]:
            x = data_size[0] - 1
        elif x < 0:
            x = 0

        if y >= data_size[1]:
            y = data_size[1]
        elif y < 0:
            y = 0

        data[x][y] += keypoints[index][2]

    data = np.clip(data, a_min=a_min, a_max=a_max)
    return data


# def getSiftKeypoints(image, threshold=0.00):
#     img_unit8 = img_as_ubyte(image)
#
#     sift = cv2.xfeatures2d.SIFT_create()
#
#     max_cnt = 100
#     while max_cnt >= 0:
#         max_cnt = max_cnt - 1
#         kp, des = sift.detectAndCompute(img_unit8, None)
#         if len(kp) != 0:
#             break
#
#     if not kp:
#         keypoints = []
#         return keypoints
#     # FILTER RESPONSES:
#     # 返回一个具体的响应值
#     responses = []
#     for x in kp:
#         responses.append(x.response)
#     responses.sort()
#
#     keypoints = []
#     index_tracker = 0
#     for x in kp:
#         if x.response >= threshold:
#             keypoints.append((x, des[index_tracker], x.response))
#         index_tracker = index_tracker + 1
#
#     # 根据response值排序，由高到低
#     keypoints = sorted(keypoints, key=lambda tup: tup[2])
#
#     return keypoints


#  随机减少字节
def mutate_erase_bytes(data):
    """
    随机删除字节
    :param data:
    :return:
    """
    num, location = list_to_byte(data)
    if len(num) == 0:
        return data
    idx = random.randrange(len(num))
    num = num[idx:random.randrange(idx, len(num))]
    data = byte_to_list(data, num, location)
    return data


# 随机插入字节
def mutate_insert_bytes(data):
    """
    随机删除字节
    :param data:
    :return:
    """
    num, location = list_to_byte(data)
    if len(num) == 0:
        return data
    idx = random.randrange(len(num))
    new_bytes = get_random_bytes(random.randrange(1, 5))
    num = num[:idx] + new_bytes + num[idx:]
    data = byte_to_list(data, num, location)
    return data


# 插入重复字节
def mutate_insert_repeated_bytes(data):
    """
    插入重复字节
    :param data:
    :return:
    """
    num, location = list_to_byte(data)
    if len(num) == 0:
        return data
    num = bytearray(num)
    idx = random.randrange(len(num))
    new_byte = get_random_byte()
    sz = random.randrange(5)
    num[idx:idx + sz] = bytes(new_byte) * sz
    num = bytes(num)
    data = byte_to_list(data, num, location)
    return data


def get_random_bytes(size):
    return bytearray(random.getrandbits(8) for _ in range(size))


def get_random_byte():
    return random.getrandbits(8)


# 随机改变字节
def mutate_change_byte(data):
    """
    随机改变字节
    :param data:
    :return:
    """
    num, location = list_to_byte(data)
    if len(num) == 0:
        return data
    num = bytearray(num)
    idx = random.randrange(len(num))
    byte = get_random_byte()
    num[idx] = byte
    num = bytes(num)
    data = byte_to_list(data, num, location)
    return data


# 改变bit
def mutate_change_bit(data):
    """
    改变bit
    :param data:
    :return:
    """
    num, location = list_to_byte(data)
    num = bytearray(num)
    if len(num) > 0:
        idx = random.randrange(len(num))
        num[idx] ^= 1 << random.randrange(8)
        num = bytes(num)
        data = byte_to_list(data, num, location)
    return data

def mutate_insert_bit(data):
    num, location = list_to_byte(data)
    num = bytearray(num)
    if len(num) > 0:
        idx = random.randrange(len(num))
        num=num[:idx]+num[idx:idx+1]+num[idx:]
        num = bytes(num)
        data = byte_to_list(data, num, location)
    return data


def mutate_erase_bit(data):
    num, location = list_to_byte(data)
    num = bytearray(num)
    if len(num) > 0:
        idx = random.randrange(len(num))
        num = num[idx:random.randrange(idx, len(num))]
        num = bytes(num)
        data = byte_to_list(data, num, location)
    return data



def mutate_change_ascii_integer(data):
    """
    :param data:
    :return:
    """
    num, location = list_to_byte(data)
    num = bytearray(num)
    start = random.randrange(len(num))
    while start < len(num) and chr(num[start]) not in string.digits:
        start += 1
    if start == len(num):
        return bytes(num)

    end = start
    while end < len(num) and chr(num[end]) in string.digits:
        end += 1

    value = int(num[start:end])
    choice = random.randrange(5)
    if choice == 0:
        value += 1
    elif choice == 1:
        value -= 1
    elif choice == 2:
        value //= 2
    elif choice == 3:
        value *= 2
    elif choice == 4:
        value *= value
        value = max(1, value)
        value = random.randrange(value)
    else:
        assert False

    to_insert = bytes(str(value), encoding='ascii')
    num[start:end] = to_insert
    num = bytes(num)
    byte_to_list(data, num, location)
    return data


def mutate_white_noise(data, a_min=0.0, a_max=1.0, sigma=5):
    """
    添加白噪音
    :param data: 待变异数据
    :param a_min: 噪声下界
    :param a_max: 噪声上界
    :return: 变异后值
    """

    data_size = []

    data_size.append(len(data))
    element = data[0]

    data_size.append(len(element))
    element = element[0]

    data_size.append(len(element))

    noise = np.random.normal(size=data_size, scale=sigma)

    mutated_image = noise + data

    mutated_image = np.clip(mutated_image, a_min=a_min, a_max=a_max)
    return mutated_image


def mutate_rotate(data, angle=45, scale=1.0):
    (h, w) = len(data), len(data[0])
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    data = np.array(data)
    rotated = cv2.warpAffine(data, M, (w, h))  # 13
    # cv2.imshow("Rotated by 45 Degrees", rotated)  # 14
    return rotated


def mutate_triangular_matrix(data, style=1):
    """
    将data值直接转化为三角矩阵
    :param data: 待转化值
    :param style: 0或其他值为上三角， 1为下三角
    :return: 三角矩阵
    """
    data = np.array(data)
    (h, w) = data.shape[0], data.shape[1]
    if style == 1:
        for i in range(h):
            for j in range(i + 1, h):
                data[i][j][0] = data[i][j][1] = data[i][j][2] = 0
    else:
        for i in range(1, h):
            for j in range(0, i):
                data[i][j][0] = data[i][j][1] = data[i][j][2] = 0
    return data


def mutate_kernel_matrix(data, kernel=([1, 1, 1], [0, 1, 0], [1, 1, 0])):
    data = np.array(data)
    (h_data, w_data) = data.shape[0], data.shape[1]
    kernel = np.array(kernel)
    (h_kernel, w_kernel) = kernel.shape[0], kernel.shape[1]
    (h_move, w_move) = h_data // h_kernel, w_data // w_kernel
    for i in range(h_move):
        for j in range(h_move):
            for channel in range(data.shape[2]):
                temp_array = data[i * h_kernel:i * h_kernel + h_kernel, j * w_kernel: j * w_kernel + w_kernel, channel]
                temp_array = np.dot(temp_array, kernel)
                data[i * h_kernel:i * h_kernel + h_kernel, j * w_kernel: j * w_kernel + w_kernel, channel] = temp_array

    return data


def mutate_scale(data, scale=0.5):
    data = np.array(data)

    # 指定fx, fy缩放比例的缩放方式
    if isinstance(scale, list):
        data = cv2.resize(data, None, fx=scale[0], fy=scale[1], interpolation=cv2.INTER_CUBIC)
    elif isinstance(scale, float) or isinstance(scale, int):
        (h, w) = (28,28)
        data = cv2.resize(data, (int(scale * h), int(scale * w)), interpolation=cv2.INTER_CUBIC)
        print("bianyihou")
        print(data.shape)
    # cv2.imshow("Rotated by 45 Degrees", data)  # 14
    return data


def mutate_precision(data, precision=10, style=0):
    """
    精度变化算子，该算子只用于降低精度，不用于提高精度
    :param data: 待调整数据
    :param precision: 精度调整值
    :param style: 精度调整方式，0为全部调整，否则调整style个
    :return: 精度调整后数据
    """
    if style < 0:
        return data
    if not style:
        data = np.array(data)
        # print(data)
        new_data = [[[round(element, precision) for element in j] for j in i] for i in data]
        new_data = np.array(new_data)
        return new_data
    else:
        (h, w, c) = len(data), len(data[0]), len(data[0][0])
        if style >= h or style >= w:
            return data
        x = random.sample(range(0, h), style)
        y = random.sample(range(0, w), style)
        z = np.random.randint(0, c, style)
        for index in range(style):
            indexX = x[index]
            indexY = y[index]
            indexZ = z[index]
            # print(data[indexX][indexY][indexZ])
            data[indexX][indexY][indexZ] = round(data[indexX][indexY][indexZ], precision)
            # print(data[indexX][indexY][indexZ])
        return data


def mutate_insert_number(data, number_tpye, insert_function):
    """
        进行异常值、极值插入的入口函数
        :param data: 待转化值
        :param number_tpye: 0-Nan, 1-inf, 2-highNumber, 3-highPrecision
        :param insert_function: 插入策略
        :return: 插入后的数值
    """
    data = np.array(data)
    locations = insert_function(data)
    if number_tpye == 0:
        numbers = [np.nan]
    elif number_tpye == 1:
        numbers = [np.inf, -np.inf]
    elif number_tpye == 2:
        numbers = [sys.float_info.max]
    elif number_tpye == 3:
        numbers = [sys.float_info.min]
    else:
        numbers = None

    if numbers is not None:
        for location in locations:
            data[location[0]][location[1]][0] = \
                data[location[0]][location[1]][1] = \
                data[location[0]][location[1]][2] = random.choice(numbers)

    return data


def triangular_insert(data):
    """
        进行三角数值插入，待调整
        :param data: 待转化值
        :return: 插入位置列表
    """
    (h, w) = data.shape[0], data.shape[1]
    return [[i, j] for i in range(h) for j in range(i + 1, h)]


def window_insert(data):
    """
        进行窗口边缘插入，待调整
        :param data: 待转化值
        :return: 插入位置列表
    """
    (h, w) = data.shape[0], data.shape[1]
    return [[i, 0] for i in range(h)] + [[0, i] for i in range(w)] + \
           [[i, w - 1] for i in range(h)] + [[h - 1, i] for i in range(w)]


# def kernel_insert(data):
#     (h, w) = data.shape[0], data.shape[1]
#     kernel = ([1, 1, 1], [0, 1, 0], [1, 1, 0])

def mutate_nan(data, insert_function=window_insert):
    return mutate_insert_number(data, 0, insert_function)


def mutate_inf(data, insert_function=window_insert):
    return mutate_insert_number(data, 1, insert_function)


def mutate_high_number(data, insert_function=triangular_insert):
    return mutate_insert_number(data, 2, insert_function)


def mutate_high_precision(data, insert_function=triangular_insert):
    return mutate_insert_number(data, 3, insert_function)


if __name__ == '__main__':
    data = np.random.rand(2,2,3)
    #data = mutate_kernel_matrix(data)
    data=mutate_precision(data)
    print(data)
