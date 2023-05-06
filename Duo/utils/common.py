def _init():
    global function_list
    global temp_list
    global mutate_sum
    global fail_list

    function_list = []
    temp_list = []
    fail_list = []
    mutate_sum = [0]


def _append():
    for item in temp_list:
        function_list.append(item)
    # print('len_temp_list:', len(temp_list))
    temp_list.clear()


def _clear():
    for item in temp_list:
        fail_list.append(item)
    temp_list.clear()


def _print():
    print(function_list)
    print(temp_list)


def _update(val):
    temp_list.append(val)


def _submit():
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0

    sum = len(function_list)
    for idx in range(sum):
        if function_list[idx] == 0:
            sum1 += 1
        elif function_list[idx] == 1:
            sum2 += 1
        elif function_list[idx] == 2:
            sum3 += 1
        elif function_list[idx] == 3:
            sum4 += 1
        else:
            sum5 += 1

    print(sum1)
    print(sum2)
    print(sum3)
    print(sum4)
    print(sum5)

    print(mutate_sum)
    print(len(function_list))
    print(len(fail_list))

    # print("1:", sum1 / sum)
    # print("2:", sum2 / sum)
    # print("3:", sum3 / sum)
    # print("4:", sum4 / sum)
    # print("5:", sum5 / sum)


def _append_mutate_sum():
    mutate_sum[0] += 5
