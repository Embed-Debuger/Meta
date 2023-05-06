import numpy as np
import MNN
import torch.nn.functional as F
import torch
F_mnn = MNN.expr

#numpy->ndarray mx.nd.array()  ndarray->numpy: D.asnumpy()

def mnn_compute_single(im, target_interface, GPU_mode=2):
    # 采用分析模式0:GPU;1:CPU-GPU对比模式;2:CPU
    input_mnn_value = None
    output_mnn_value = None
    input_mnn_cpu_value = None
    output_mnn_cpu_value = None
    mnn_shape1 = [1]
    mnn_shape = [0 for _ in range(4)]


    for shape_element in im.shape:
        mnn_shape1.append(shape_element)

    input_pytorch_cpu_value = im.reshape(mnn_shape1)
    pytorch_out = F.max_pool2d(torch.from_numpy(np.rollaxis(input_pytorch_cpu_value, 3, 1)), kernel_size=(2, 2),
                               stride=(2, 2)).numpy()

    mnn_shape[0]=mnn_shape1[0]
    mnn_shape[1]=mnn_shape1[3]
    mnn_shape[2]=mnn_shape1[1]
    mnn_shape[3]=mnn_shape1[2]

    # input_mnn = F_mnn.const(im.flatten().tolist(), mnn_shape, F_mnn.data_format.NCHW)
    input_mnn = F_mnn.const(im.flatten().tolist(), mnn_shape)

    if GPU_mode != 0:
    #if !=-0
        input_mnn_cpu_value = np.array(input_mnn.read()).transpose(0,2,3,1)

        #print(input_mnn_cpu_value)
        if target_interface == 'conv1':
            input_mnn = F_mnn.convert(input_mnn, F_mnn.NC4HW4)  # 卷积层需要这么设置
            weights_mnn = F_mnn.const(np.full((11, 11, 3, 1), 1, np.float32).transpose((3, 2, 0, 1)),[1,3,11,11])

            bias_mx = F_mnn.const(np.zeros((3),dtype=np.float32),[1,3])

            output_mnn_cpu=F_mnn.conv2d(input=input_mnn,weight=weights_mnn,bias=bias_mx,stride=[4,4],padding=[0,0]).read()
            output_mnn_cpu=np.array(output_mnn_cpu).transpose((0, 2, 3, 1))
            # print(output_mnn_cpu.shape)

        elif target_interface == 'pool1':
            output_mnn_cpu=F_mnn.max_pool(input_mnn,kernel=[2,2],stride=[2,2]).read()

            output_mnn_cpu=np.array(output_mnn_cpu).reshape(pytorch_out.shape)
            output_mnn_cpu=output_mnn_cpu.transpose(0,2,3,1)
            # print(output_mnn_cpu.shape)

        elif target_interface == 'pool2':
            output_mnn_cpu = F_mnn.avg_pool(input_mnn, kernel=[2, 2], stride=[2, 2]).read()
            output_mnn_cpu = np.array(output_mnn_cpu).reshape(pytorch_out.shape)
            output_mnn_cpu = output_mnn_cpu.transpose(0, 2, 3, 1)
            # print(output_mnn_cpu)


        elif target_interface == 'relu1':
            output_mnn_cpu = F_mnn.relu(input_mnn).read()
            output_mnn_cpu = np.array(output_mnn_cpu).reshape(mnn_shape).transpose(0,2,3,1)
            #print(output_mnn_cpu)

        #elif target_interface == 'dense1':


        elif target_interface == 'sigmoid1':
            output_mnn_cpu = F_mnn.sigmoid(input_mnn).read()
            output_mnn_cpu=np.array(output_mnn_cpu).reshape(mnn_shape).transpose(0,2,3,1)
            #print(output_mnn_cpu.shape)

        elif target_interface == 'tanh1':
            output_mnn_cpu = F_mnn.tanh(input_mnn).read()
            output_mnn_cpu=np.array(output_mnn_cpu).reshape(mnn_shape).transpose(0,2,3,1)

        elif target_interface == 'softmax1':
            output_mnn_cpu = F_mnn.softmax(input_mnn).read()
            output_mnn_cpu = np.array(output_mnn_cpu).reshape(mnn_shape).transpose(0,2,3,1)

        elif target_interface == 'norm1':
             mnn_norm = MNN.nn.batch_norm(3)
             output_mnn_cpu = mnn_norm(input_mnn).read()
             output_mnn_cpu = np.array(output_mnn_cpu).reshape(mnn_shape).transpose(0, 2, 3, 1)
             #print(output_mnn_cpu.shape)

        else:
            output_mnn_cpu = None
        output_mnn_cpu_value = output_mnn_cpu

    if GPU_mode != 2:
        input_mnn_value = input_mnn.to("cuda").to("cpu").numpy()
        if target_interface == 'conv1':
            weights_torch = torch.from_numpy(np.full((11, 11, 3, 1), 1, np.float64).transpose((3, 2, 0, 1)))
            output_mnn = torch.from_numpy(
                F.conv2d(torch.from_numpy(input_mnn.numpy().transpose((0, 3, 1, 2))).to("cuda"),
                         weights_torch.to("cuda"), padding=0, stride=4).to("cpu").numpy().transpose((0, 2, 3, 1))).to(
                "cuda")
        elif target_interface == 'conv2':
            x_torch = torch.from_numpy(input_mnn.numpy().transpose((0, 3, 1, 2)).astype(np.float64))
            weights_torch = torch.from_numpy(np.full((11, 11, 3, 1), 1, np.float64).transpose((3, 2, 0, 1)))
            stride = 4
            if x_torch.numpy().shape[2] % stride == 0:
                pad = max(weights_torch.numpy().shape[2] - stride, 0)
            else:
                pad = max(weights_torch.numpy().shape[2] - (x_torch.numpy().shape[2] % stride), 0)

            if pad % 2 == 0:
                pad_val = pad // 2
                padding = (pad_val, pad_val, pad_val, pad_val)
            else:
                pad_val_start = pad // 2
                pad_val_end = pad - pad_val_start
                padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
            x_torch = F.pad(x_torch, padding, "constant", 0)
            output_mnn = torch.from_numpy(
                F.conv2d(x_torch.to("cuda"), weights_torch.to("cuda"), padding=0, stride=4).to("cpu").numpy().transpose(
                    (0, 2, 3, 1))).to("cuda")
        elif target_interface == 'pool1':
            output_mnn = torch.from_numpy(np.rollaxis(
                F.max_pool2d(torch.from_numpy(np.rollaxis(input_mnn_value, 3, 1)).to("cuda"), kernel_size=(2, 2),
                             stride=(2, 2)).to("cpu").numpy(), 1, 4)).to("cuda")
        elif target_interface == 'pool2':
            output_mnn = torch.from_numpy(np.rollaxis(
                F.avg_pool2d(torch.from_numpy(np.rollaxis(input_mnn_value, 3, 1)).to("cuda"), kernel_size=(2, 2),
                             stride=(2, 2)).to("cpu").numpy(), 1, 4)).to("cuda")
        elif target_interface == 'relu1':
            output_mnn = F.relu(input_mnn.to("cuda"))
        elif target_interface == 'dense1':
            output_mnn = None
        elif target_interface == 'sigmoid1':
            output_mnn = F.sigmoid(input_mnn.to("cuda"))
        elif target_interface == 'tanh1':
            output_mnn = F.tanh(input_mnn.to("cuda"))
        else:
            output_mnn = None
        output_mnn_value = output_mnn.to("cpu").numpy()

    return output_mnn_value, input_mnn_value, output_mnn_cpu_value, input_mnn_cpu_value


# if __name__=='__main__':
#     a=np.ones([12,12,3])
#     mnn_compute_single(a,"norm1",2)
