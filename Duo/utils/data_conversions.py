import struct


# def int_to_byte(li):
#     for i in range(len(li)):
#         for j in range(len(li[0])):
#             for k in range(len(li[0][0])):
#                 # print(li[i][j][k])
#                 # print(bytes(li[i][j][k]))
#                 li[i][j][k] = bytes([li[i][j][k]])
#     return li
def int_to_byte(data):
    return bytes([data])


# def byte_to_int(li):
#     for i in range(len(li)):
#         for j in range(len(li[0])):
#             for k in range(len(li[0][0])):
#                 # print(li[i][j][k])
#                 # print(bytes(li[i][j][k]))
#                 li[i][j][k] = int.from_bytes(li[i][j][k], byteorder='little', signed=True)
#     return li
def byte_to_int(data):
    return int.from_bytes(data, byteorder='little', signed=True)


# def float_to_byte(li):
#     for i in range(len(li)):
#         for j in range(len(li[0])):
#             for k in range(len(li[0][0])):
#                 li[i][j][k] = struct.pack('>f', li[i][j][k])
#     return li
def float_to_byte(data):
    return struct.pack('>f', data)


# def byte_to_float(li):
#     for i in range(len(li)):
#         for j in range(len(li[0])):
#             for k in range(len(li[0][0])):
#                 li[i][j][k] = struct.unpack('>f', li[i][j][k])[0]
#     return li
def byte_to_float(data):
    return struct.unpack('>f', data)[0]


# def double_to_byte(li):
#     for i in range(len(li)):
#         for j in range(len(li[0])):
#             for k in range(len(li[0][0])):
#                 li[i][j][k] = struct.pack('>d', li[i][j][k])
#     return li
def double_to_byte(data):
    return struct.pack('>d', data)


# def byte_to_double(li):
#     for i in range(len(li)):
#         for j in range(len(li[0])):
#             for k in range(len(li[0][0])):
#                 li[i][j][k] = struct.unpack('>d', li[i][j][k])[0]
#     return li
def byte_to_double(data):
    return struct.unpack('>d', data)[0]


# def converse(li, target):
#     for i in range(len(li)):
#         for j in range(len(li[0])):
#             for k in range(len(li[0][0])):
#                 li[i][j][k] = target(li[i][j][k])
#     return li

def converse(li, target):
    return target(li)
