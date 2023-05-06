import numpy
import os

path = './results/pool1_0_1'
if os.path.exists(path):
    ls = os.listdir(path)
    ls_crash = os.listdir(path+'/crashes')
    counter_crash = 0
    counter_data = 0
    counter_susccess = 0
    for element in ls:
        if 'data' in element:
            counter_data += 1
        if 'input_caffe' in element:
            counter_susccess += 1
    for element in ls_crash:
        if 'data' in element:
            counter_crash += 1
    counter_data //= 3
    counter_susccess //= 3
    counter_crash //= 3
    print(counter_data)
    print(counter_susccess)
    print(counter_crash)




