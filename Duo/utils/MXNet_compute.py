import numpy as np
import mxnet as mx
from  mxnet.gluon import nn
from mxnet import nd

#numpy->ndarray mx.nd.array()  ndarray->numpy: D.asnumpy()

def mxnet_compute_single(im, target_interface, GPU_mode=1):
    # 采用分析模式0:GPU;1:CPU-GPU对比模式;2:CPU
    input_mxnet_value = None
    output_mxnet_value = None
    input_mxnet_cpu_value = None
    output_mxnet_cpu_value = None
    mxnet_shape = [1]
    #mxnet_shape = []
    for shape_element in im.shape:
        mxnet_shape.append(shape_element)
    #input_mxnet = mx..reshape(torch.from_numpy(im), mxnet_shape)
    input_mxnet = mx.nd.Reshape(mx.nd.array(im),mxnet_shape)
    #print(mxnet_shape)
    print(input_mxnet.shape)
    # print(input_mxnet.asnumpy())

    if GPU_mode == 2:
    #if !=-0
        input_mxnet_cpu_value = input_mxnet.asnumpy()

        if target_interface == 'conv1':
            weights_mx = mx.nd.array(np.full((11, 11, 3, 1), 1, np.float64).transpose((3, 2, 0, 1)))
            bias_mx = mx.nd.array(np.full((1,), 0, np.float64))
            input=input_mxnet.asnumpy().transpose((0, 3, 1, 2))

            output_mxnet_cpu=nd.array(mx.ndarray.op.Convolution(nd.array(input),weights_mx,bias=bias_mx,pad=(0,0),stride=(4,4),kernel=(11,11),num_filter=1).asnumpy().transpose((0, 2, 3, 1)))
            print(output_mxnet_cpu.shape)

        elif target_interface == 'pool1':
            # output_mxnet_cpu = torch.from_numpy(np.rollaxis(
            #     F.max_pool2d(torch.from_numpy(np.rollaxis(input_mxnet_cpu_value, 3, 1)), kernel_size=(2, 2),
            #                  stride=(2, 2)).numpy(), 1, 4))
            maxpool=nn.MaxPool2D(pool_size=(2,2),strides=(2,2))
            output_mxnet_cpu=nd.array(np.rollaxis(maxpool(nd.array(np.rollaxis(input_mxnet_cpu_value,3,1))).asnumpy(),1,4))
            print(output_mxnet_cpu.shape)

        elif target_interface == 'pool2':
            avgpool = nn.AvgPool2D(pool_size=(2, 2), strides=(2, 2))
            output_mxnet_cpu = nd.array(
                np.rollaxis(avgpool(nd.array(np.rollaxis(input_mxnet_cpu_value, 3, 1))).asnumpy(), 1, 4))
            #print(output_mxnet_cpu.shape)

        elif target_interface == 'relu1':
            output_mxnet_cpu = nd.ndarray.op.relu(input_mxnet)
            #print(output_mxnet_cpu)

        elif target_interface == 'dense1':
            #output_mxnet_cpu = None
            weight=[[1.0] for j in range(0)]
            # for i in range(9):
            #     weight[i][0]=1.0
            init=mx.initializer.Constant(0.001)
            dense=nn.Dense(units=1,weight_initializer=init)
            dense.initialize()
            output_mxnet_cpu=dense(input_mxnet)
            #print(output_mxnet_cpu)

        elif target_interface == 'sigmoid1':
            output_mxnet_cpu = mx.ndarray.op.sigmoid(input_mxnet)
           # print(output_mxnet_cpu)

        elif target_interface == 'tanh1':
            output_mxnet_cpu = mx.ndarray.op.tanh(input_mxnet)
            #print(output_mxnet_cpu)

        elif target_interface == 'softmax1':
            output_mxnet_cpu= mx.ndarray.op.softmax(input_mxnet)
            #print(output_mxnet_cpu)

        elif target_interface == "norm1":
            batch_norm=nn.BatchNorm(3,momentum=0.99)
            batch_norm.initialize()
            output_mxnet_cpu=batch_norm(input_mxnet)
            print(output_mxnet_cpu.shape)
        else:
            output_mxnet_cpu = None
        output_mxnet_cpu_value = output_mxnet_cpu.asnumpy()

    if GPU_mode == 0:
        output_mxnet=None
        mxnet_shape = [1]
        # mxnet_shape = []
        for shape_element in im.shape:
            mxnet_shape.append(shape_element)
        # input_mxnet = mx..reshape(torch.from_numpy(im), mxnet_shape)
        input_mxnet = mx.nd.Reshape(mx.nd.array(im,ctx=mx.gpu()), mxnet_shape)
        # print(mxnet_shape)
        # print(input_mxnet)
        # print(input_mxnet.asnumpy())


        # if !=-0
        input_mxnet_value = input_mxnet.asnumpy()

        if target_interface == 'conv1':
            weights_mx = mx.nd.array(np.full((11, 11, 3, 1), 1, np.float64).transpose((3, 2, 0, 1)),ctx=mx.gpu())
            bias_mx = mx.nd.array(np.full((1,), 0, np.float64),ctx=mx.gpu())
            input = input_mxnet.asnumpy().transpose((0, 3, 1, 2))

            output_mxnet = nd.array(
                mx.ndarray.op.Convolution(nd.array(input,ctx=mx.gpu()), weights_mx, bias=bias_mx, pad=(0, 0), stride=(4, 4),
                                          kernel=(11, 11), num_filter=1).asnumpy().transpose((0, 2, 3, 1)))
            print("conv1_gpu")
            print(output_mxnet.shape)


        elif target_interface == 'pool1':
            # output_mxnet_cpu = torch.from_numpy(np.rollaxis(
            #     F.max_pool2d(torch.from_numpy(np.rollaxis(input_mxnet_cpu_value, 3, 1)), kernel_size=(2, 2),
            #                  stride=(2, 2)).numpy(), 1, 4))
            maxpool = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
            output_mxnet = nd.array(
                np.rollaxis(maxpool(nd.array(np.rollaxis(input_mxnet_value, 3, 1),ctx=mx.gpu())).asnumpy(), 1, 4),ctx=mx.gpu())
            print(output_mxnet.shape)

        elif target_interface == 'pool2':
            avgpool = nn.AvgPool2D(pool_size=(2, 2), strides=(2, 2))
            output_mxnet = nd.array(
                np.rollaxis(avgpool(nd.array(np.rollaxis(input_mxnet_value, 3, 1),ctx=mx.gpu())).asnumpy(), 1, 4),ctx=mx.gpu())
            # print(output_mxnet_cpu.shape)

        elif target_interface == 'relu1':
            output_mxnet = nd.ndarray.op.relu(input_mxnet)
            # print(output_mxnet_cpu)

        elif target_interface == 'dense1':
            # output_mxnet_cpu = None
            weight = [[1.0] for j in range(0)]
            # for i in range(9):
            #     weight[i][0]=1.0
            init = mx.initializer.Constant(0.001)
            dense = nn.Dense(units=1, weight_initializer=init)
            dense.initialize(ctx=mx.gpu())
            output_mxnet = dense(input_mxnet)
            # print(output_mxnet_cpu)

        elif target_interface == 'sigmoid1':
            output_mxnet = mx.ndarray.op.sigmoid(input_mxnet)
        # print(output_mxnet_cpu)

        elif target_interface == 'tanh1':
            output_mxnet = mx.ndarray.op.tanh(input_mxnet)
            # print(output_mxnet_cpu)

        elif target_interface == 'softmax1':
            output_mxnet = mx.ndarray.op.softmax(input_mxnet)
            # print(output_mxnet_cpu)

        elif target_interface == "norm1":
            batch_norm = nn.BatchNorm(3, momentum=0.99)
            batch_norm.initialize(ctx=mx.gpu())
            output_mxnet = batch_norm(input_mxnet)
            print(output_mxnet.shape)
        else:
            output_mxnet = None

        output_mxnet_value = output_mxnet.asnumpy()


    return output_mxnet_value, input_mxnet_value, output_mxnet_cpu_value, input_mxnet_cpu_value



# if __name__=='__main__':
#     a=np.ones([64,64,3])
#     mxnet_compute_single(a,"pool1",2)