import numpy as np
import keras

def BnModel(dshape):
    x = keras.layers.Input(shape=dshape[1:])
    y = keras.layers.BatchNormalization(axis=1)(x)
    return keras.models.Model(x,y)
data_shape = [10,3,32,32]
data = np.random.uniform(-1,1,data_shape)
model = BnModel(data_shape)
result = model.predict(data)


