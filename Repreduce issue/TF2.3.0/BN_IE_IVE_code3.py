from tensorflow.keras.layers import BatchNormalization, Input
from tensorflow.keras.models import Model, clone_model
# from tensorflow._api.v1.keras.layers import BatchNormalization, Input
# from tensorflow._api.v1.keras.models import Model, clone_model

import os
import re
import numpy as np

def SourceModel(shape):
    x = Input(shape=shape[1:])
    y = BatchNormalization(axis=-1)(x)
    return Model(x, y)

def FollowModel_1(source_model):
    follow_model = clone_model(source_model)
    # 读取参数
    weights = source_model.get_weights()
    weights_names = [weight.name for layer in source_model.layers for weight in layer.weights]
    variance_idx = FindWeightsIdx("variance", weights_names)

    # 变异模型
    # delta = np.random.uniform(-1e-3, 1e-3, 1)[0]
    follow_model.layers[1].epsilon += delta     # 写BN层的epsilon
    weights[variance_idx] -= delta
    follow_model.set_weights(weights)

    return follow_model

def FindWeightsIdx(name, weights_names):
    # find layer index by name
    for idx, names in enumerate(weights_names):
        if re.search(name, names):
            return idx
    return -1


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

shape = (10, 32, 32, 3)
data = np.random.uniform(-1, 1, shape)
delta = -1

source_model = SourceModel(shape)
follow_model = FollowModel_1(source_model)

source_result = source_model.predict(data)
follow_result = follow_model.predict(data)
dis = np.sum(abs(source_result-follow_result))

print("delta:", delta, "; dis:", dis)


# https://github.com/tensorflow/tensorflow/issues/59309
# https://stackoverflow.com/questions/75350569/a-batch-normalization-error-in-the-epsilon-of-tensorflow


