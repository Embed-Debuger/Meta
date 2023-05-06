import os
import numpy as np
import torch
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
def pytorch_compute_single(im, target_interface, GPU_mode=1):
    input_pytorch_value = None
    output_pytorch_value = None
    input_pytorch_cpu_value = None
    output_pytorch_cpu_value = None
    pytorch_shape = [1]
    for shape_element in im.shape:
        pytorch_shape.append(shape_element)

    #print(pytorch_shape)
    input_pytorch = torch.reshape(torch.from_numpy(im), pytorch_shape)
    if GPU_mode != 0:
        input_pytorch_cpu_value = input_pytorch.numpy()

        if target_interface == 'conv1':
            weights_torch = torch.from_numpy(np.full((11, 11, 3, 1), 1, np.float64).transpose((3, 2, 0, 1)))
            output_pytorch_cpu = torch.from_numpy(
                F.conv2d(torch.from_numpy(input_pytorch.numpy().transpose((0, 3, 1, 2))), weights_torch, padding=0,
                         stride=4).numpy().transpose((0, 2, 3, 1)))
        elif target_interface == 'conv2':
            x_torch = torch.from_numpy(input_pytorch.numpy().transpose((0, 3, 1, 2)).astype(np.float64))
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
            output_pytorch_cpu = torch.from_numpy(
                F.conv2d(x_torch, weights_torch, padding=0, stride=stride).numpy().transpose((0, 2, 3, 1)))

        elif target_interface == 'pool1':
            output_pytorch_cpu = torch.from_numpy(np.rollaxis(
                F.max_pool2d(torch.from_numpy(np.rollaxis(input_pytorch_cpu_value, 3, 1)), kernel_size=(2, 2),
                             stride=(2, 2)).numpy(), 1, 4))

        elif target_interface == 'pool2':
            output_pytorch_cpu = torch.from_numpy(np.rollaxis(
                F.avg_pool2d(torch.from_numpy(np.rollaxis(input_pytorch_cpu_value, 3, 1)), kernel_size=(2, 2),
                             stride=(2, 2)).numpy(), 1, 4))
        elif target_interface == 'relu1':
            output_pytorch_cpu = F.relu(input_pytorch)
        elif target_interface == 'dense1':
            output_pytorch_cpu = None
        elif target_interface == 'sigmoid1':
            output_pytorch_cpu = F.sigmoid(input_pytorch)


        elif target_interface == 'tanh1':
            output_pytorch_cpu = F.tanh(input_pytorch)

        elif target_interface == 'softmax1':
            output_pytorch_cpu =F.softmax(input_pytorch)

        elif target_interface == 'norm1':
            input_pytorch=torch.transpose(input_pytorch,1,3).to(torch.float32)
            torch_norm = torch.nn.BatchNorm2d(3, momentum=0.99)
            output_pytorch_cpu = torch_norm(input_pytorch)
            output_pytorch_cpu=torch.transpose(output_pytorch_cpu,1,3)

            #output_pytorch_cpu = F.batch_norm(input_pytorch,momentum = 0.99, eps = 1e-05)
            #print(output_pytorch_cpu.shape)
        else:
            output_pytorch_cpu = None

        if target_interface == 'norm1':
            output_pytorch_cpu_value = output_pytorch_cpu.detach().numpy()
        else:
            output_pytorch_cpu_value = output_pytorch_cpu.numpy()
        # print(output_pytorch_cpu_value.shape)

    if GPU_mode != 2:
        with torch.no_grad():
            input_pytorch_value = input_pytorch.to("cuda").numpy()
            if target_interface == 'conv1':
                weights_torch = torch.from_numpy(np.full((11, 11, 3, 1), 1, np.float64).transpose((3, 2, 0, 1)))
                output_pytorch = torch.from_numpy(
                    F.conv2d(torch.from_numpy(input_pytorch.numpy().transpose((0, 3, 1, 2))).to("cuda"),
                             weights_torch.to("cuda"), padding=0, stride=4).to("cpu:0").numpy().transpose((0, 2, 3, 1))).to(
                    "cuda")
            elif target_interface == 'conv2':
                x_torch = torch.from_numpy(input_pytorch.numpy().transpose((0, 3, 1, 2)).astype(np.float64))
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
                output_pytorch = torch.from_numpy(
                    F.conv2d(x_torch.to("cuda"), weights_torch.to("cuda"), padding=0, stride=4).to("cpu").numpy().transpose(
                        (0, 2, 3, 1))).to("cuda")


            elif target_interface == 'pool1':
                output_pytorch = torch.from_numpy(np.rollaxis(
                    F.max_pool2d(torch.from_numpy(np.rollaxis(input_pytorch_value, 3, 1)).to("cuda"), kernel_size=(2, 2),
                                 stride=(2, 2)).to("cpu").numpy(), 1, 4)).to("cuda")
            elif target_interface == 'pool2':
                output_pytorch = torch.from_numpy(np.rollaxis(
                    F.avg_pool2d(torch.from_numpy(np.rollaxis(input_pytorch_value, 3, 1)).to("cuda"), kernel_size=(2, 2),
                                 stride=(2, 2)).to("cpu").numpy(), 1, 4)).to("cuda")
            elif target_interface == 'relu1':
                output_pytorch = F.relu(input_pytorch.to("cuda"))
            elif target_interface == 'dense1':
                output_pytorch = None
            elif target_interface == 'sigmoid1':
                output_pytorch = F.sigmoid(input_pytorch.to("cuda"))
            elif target_interface == 'tanh1':
                output_pytorch = F.tanh(input_pytorch.to("cuda"))
            else:
                output_pytorch = None
            output_pytorch_value = output_pytorch.to("cpu").numpy()
    return output_pytorch_value, input_pytorch_value, output_pytorch_cpu_value, input_pytorch_cpu_value

# if __name__=='__main__':
#     a=np.ones([12,12,3])
#     pytorch_compute_single(a,"norm1",2)