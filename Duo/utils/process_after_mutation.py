import random


def process_insert_bytes(li):
    for i in range(len(li)):
        for j in range(len(li[0])):
            for k in range(len(li[0][0])):
                result = bytes()
                data = li[i][j][k]
                idx = random.randrange(len(data))
                while idx + 8 > len(data):
                    idx = random.randrange(len(data))
                for index in range(0, 8):
                    # print(start)
                    result += bytes([data[index]])
                # print(len(result))
                li[i][j][k] = result
    return li


def process(li):
    length = len(li)
    if length > 4:
        result = bytes()
        for idx in range(0, 4):
            result += bytes([li[idx]])
        li = result
    elif length < 4:
        data = bytearray(li)
        size = 4 - len(data)
        data[length: length + size] = bytes('\x00', encoding='utf-8') * size
        data = bytes(data)
        li = data

    return li
