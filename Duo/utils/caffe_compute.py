import os

os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np

path_origin = './net'


class TopologyElement(object):
    def __init__(self, name, related_interface, path, output=None):
        """
        用于存储算子组合拓扑结构信息
        :param name:结构名称,str
        :param related_interface:相关的单一算子，List(str)
        :param path:配置文件路径，包含CPU/GPU，List(str)
        :param output:对应的输出label
        """
        self.name = name
        self.related_interface = related_interface
        self.path = path
        self.output = output

    def match(self, target_interface):
        idx = 0
        for idx in range(0, len(target_interface)):
            if str.isdigit(target_interface[idx]):
                break
        pattern = target_interface[0:idx]
        if pattern in self.related_interface:
            return True
        return False

    def getPath(self):
        return self.path

    def setPath(self, path):
        self.path = path

    def getName(self):
        return self.name

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output


def save_proto(proto, prototxt):
    with open(prototxt, 'w') as f:
        f.write(str(proto))


def generate_topologies():
    # Lenet结构
    # conv_pool1
    conv_pool1 = TopologyElement('conv_pool1', ['conv', 'pool'], None)
    conv_pool1_path = path_origin + '/train_' + conv_pool1.getName() + '.prototxt'
    conv_pool1.setPath(conv_pool1_path)
    conv_pool1_n = caffe.NetSpec()

    conv_pool1_n.data = L.DummyData(shape=dict(dim=[1, 3, 28, 28]))
    conv_pool1_n.conv1 = L.Convolution(conv_pool1_n.data, kernel_size=5, num_output=20, stride=1,
                                       weight_filler=dict(type='xavier'),
                                       bias_filler=dict(type='constant'))
    conv_pool1_n.pool1 = L.Pooling(conv_pool1_n.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    save_proto(conv_pool1_n.to_proto(), conv_pool1_path)
    conv_pool1.setOutput('pool1')

    # conv_pool2
    conv_pool2 = TopologyElement('conv_pool2', ['conv', 'pool'], None)
    conv_pool2_path = path_origin + '/train_' + conv_pool2.getName() + '.prototxt'
    conv_pool2.setPath(conv_pool2_path)
    conv_pool2_n = caffe.NetSpec()

    conv_pool2_n.data = L.DummyData(shape=dict(dim=[1, 3, 28, 28]))
    conv_pool2_n.conv1 = L.Convolution(conv_pool2_n.data, kernel_size=5, num_output=50, stride=1,
                                       weight_filler=dict(type='xavier'),
                                       bias_filler=dict(type='constant'))
    conv_pool2_n.pool1 = L.Pooling(conv_pool2_n.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    save_proto(conv_pool2_n.to_proto(), conv_pool2_path)
    conv_pool2.setOutput('pool1')

    # relu_softmax1
    relu_softmax1 = TopologyElement('relu_softmax1', ['relu', 'softmax'], None)
    relu_softmax1_path = path_origin + '/train_' + relu_softmax1.getName() + '.prototxt'
    relu_softmax1.setPath(relu_softmax1_path)
    relu_softmax1_n = caffe.NetSpec()

    relu_softmax1_n.data = L.DummyData(shape=dict(dim=[1, 3, 28, 28]))
    relu_softmax1_n.relu1 = L.ReLU(relu_softmax1_n.data)

    relu_softmax1_n.softmax1 = L.Softmax(relu_softmax1_n.relu1)

    save_proto(relu_softmax1_n.to_proto(), conv_pool2_path)
    relu_softmax1.setOutput('softmax1')

    # Vgg结构
    # conv_relu_pool1
    conv_relu_pool1 = TopologyElement('conv_relu_pool1', ['conv', 'relu', 'pool'], None)
    conv_relu_pool1_path = path_origin + '/train_' + conv_relu_pool1.getName() + '.prototxt'
    conv_relu_pool1.setPath(conv_relu_pool1_path)
    conv_relu_pool1_n = caffe.NetSpec()

    conv_relu_pool1_n.data = L.DummyData(shape=dict(dim=[1, 3, 28, 28]))
    conv_relu_pool1_n.conv1 = L.Convolution(conv_relu_pool1_n.data, kernel_size=3, num_output=64, stride=1,
                                       weight_filler=dict(type='xavier'),
                                       bias_filler=dict(type='constant'))
    conv_relu_pool1_n.relu1 = L.ReLU(conv_relu_pool1_n.conv1)
    conv_relu_pool1_n.conv2 = L.Convolution(conv_relu_pool1_n.relu1, kernel_size=3, num_output=64, stride=1,
                                       weight_filler=dict(type='xavier'),
                                       bias_filler=dict(type='constant'))
    conv_relu_pool1_n.relu2 = L.ReLU(conv_relu_pool1_n.conv2)
    conv_relu_pool1_n.pool1 = L.Pooling(conv_relu_pool1_n.relu2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    save_proto(conv_relu_pool1_n.to_proto(), conv_relu_pool1_path)
    conv_relu_pool1.setOutput('pool1')

    # conv_relu_pool2
    conv_relu_pool2 = TopologyElement('conv_relu_pool2', ['conv', 'relu', 'pool'], None)
    conv_relu_pool2_path = path_origin + '/train_' + conv_relu_pool2.getName() + '.prototxt'
    conv_relu_pool2.setPath(conv_relu_pool2_path)
    conv_relu_pool2_n = caffe.NetSpec()

    conv_relu_pool2_n.data = L.DummyData(shape=dict(dim=[1, 3, 28, 28]))
    conv_relu_pool2_n.conv1 = L.Convolution(conv_relu_pool2_n.data, kernel_size=3, num_output=128, stride=1,
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant'))
    conv_relu_pool2_n.relu1 = L.ReLU(conv_relu_pool2_n.conv1)
    conv_relu_pool2_n.conv2 = L.Convolution(conv_relu_pool2_n.relu1, kernel_size=3, num_output=128, stride=1,
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant'))
    conv_relu_pool2_n.relu2 = L.ReLU(conv_relu_pool2_n.conv2)
    conv_relu_pool2_n.pool1 = L.Pooling(conv_relu_pool2_n.relu2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    save_proto(conv_relu_pool2_n.to_proto(), conv_relu_pool2_path)
    conv_relu_pool2.setOutput('pool1')

    # conv_relu_pool3
    conv_relu_pool3 = TopologyElement('conv_relu_pool3', ['conv', 'relu', 'pool'], None)
    conv_relu_pool3_path = path_origin + '/train_' + conv_relu_pool3.getName() + '.prototxt'
    conv_relu_pool3.setPath(conv_relu_pool3_path)
    conv_relu_pool3_n = caffe.NetSpec()

    conv_relu_pool3_n.data = L.DummyData(shape=dict(dim=[1, 3, 28, 28]))
    conv_relu_pool3_n.conv1 = L.Convolution(conv_relu_pool3_n.data, kernel_size=3, num_output=256, stride=1,
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant'))
    conv_relu_pool3_n.relu1 = L.ReLU(conv_relu_pool3_n.conv1)
    conv_relu_pool3_n.conv2 = L.Convolution(conv_relu_pool3_n.relu1, kernel_size=3, num_output=256, stride=1,
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant'))
    conv_relu_pool3_n.relu2 = L.ReLU(conv_relu_pool3_n.conv2)
    conv_relu_pool3_n.conv3 = L.Convolution(conv_relu_pool3_n.relu2, kernel_size=3, num_output=256, stride=1,
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant'))
    conv_relu_pool3_n.relu3 = L.ReLU(conv_relu_pool3_n.conv3)
    conv_relu_pool3_n.pool1 = L.Pooling(conv_relu_pool3_n.relu3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    save_proto(conv_relu_pool3_n.to_proto(), conv_relu_pool3_path)
    conv_relu_pool3.setOutput('pool1')

    # conv_relu_pool4
    conv_relu_pool4 = TopologyElement('conv_relu_pool4', ['conv', 'relu', 'pool'], None)
    conv_relu_pool4_path = path_origin + '/train_' + conv_relu_pool4.getName() + '.prototxt'
    conv_relu_pool4.setPath(conv_relu_pool4_path)
    conv_relu_pool4_n = caffe.NetSpec()

    conv_relu_pool4_n.data = L.DummyData(shape=dict(dim=[1, 3, 28, 28]))
    conv_relu_pool4_n.conv1 = L.Convolution(conv_relu_pool4_n.data, kernel_size=3, num_output=256, stride=1,
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant'))
    conv_relu_pool4_n.relu1 = L.ReLU(conv_relu_pool4_n.conv1)
    conv_relu_pool4_n.conv2 = L.Convolution(conv_relu_pool4_n.relu1, kernel_size=3, num_output=256, stride=1,
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant'))
    conv_relu_pool4_n.relu2 = L.ReLU(conv_relu_pool4_n.conv2)
    conv_relu_pool4_n.conv3 = L.Convolution(conv_relu_pool4_n.relu2, kernel_size=3, num_output=256, stride=1,
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant'))
    conv_relu_pool4_n.relu3 = L.ReLU(conv_relu_pool4_n.conv3)
    conv_relu_pool4_n.conv4 = L.Convolution(conv_relu_pool4_n.relu3, kernel_size=3, num_output=256, stride=1,
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant'))
    conv_relu_pool4_n.relu4 = L.ReLU(conv_relu_pool4_n.conv4)
    conv_relu_pool4_n.pool1 = L.Pooling(conv_relu_pool4_n.relu4, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    save_proto(conv_relu_pool4_n.to_proto(), conv_relu_pool4_path)
    conv_relu_pool4.setOutput('pool1')

    # SE-BN-Inception
    # conv_sigmoid1
    conv_sigmoid1 = TopologyElement('conv_sigmoid1', ['conv', 'sigmoid'], None)
    conv_sigmoid1_path = path_origin + '/train_' + conv_sigmoid1.getName() + '.prototxt'
    conv_sigmoid1.setPath(conv_sigmoid1_path)
    conv_sigmoid1_n = caffe.NetSpec()

    conv_sigmoid1_n.data = L.DummyData(shape=dict(dim=[1, 3, 28, 28]))
    conv_sigmoid1_n.conv1 = L.Convolution(conv_sigmoid1_n.data, kernel_size=1, num_output=320, stride=1,
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant'))
    conv_sigmoid1_n.sigmoid1 = L.Sigmoid(conv_sigmoid1_n.conv1)

    conv_sigmoid1.setOutput('sigmoid1')

    # SE-ResNet-50
    # conv_sigmoid2
    conv_sigmoid2 = TopologyElement('conv_sigmoid2', ['conv', 'sigmoid'], None)
    conv_sigmoid2_path = path_origin + '/train_' + conv_sigmoid2.getName() + '.prototxt'
    conv_sigmoid2.setPath(conv_sigmoid2_path)
    conv_sigmoid2_n = caffe.NetSpec()

    conv_sigmoid2_n.data = L.DummyData(shape=dict(dim=[1, 3, 28, 28]))
    conv_sigmoid2_n.conv1 = L.Convolution(conv_sigmoid2_n.data, kernel_size=1, num_output=256, stride=1,
                                          weight_filler=dict(type='xavier'),
                                          bias_filler=dict(type='constant'))
    conv_sigmoid2_n.sigmoid1 = L.Sigmoid(conv_sigmoid2_n.conv1)

    conv_sigmoid2.setOutput('sigmoid1')

    # conv_sigmoid3
    conv_sigmoid3 = TopologyElement('conv_sigmoid3', ['conv', 'sigmoid'], None)
    conv_sigmoid3_path = path_origin + '/train_' + conv_sigmoid3.getName() + '.prototxt'
    conv_sigmoid3.setPath(conv_sigmoid3_path)
    conv_sigmoid3_n = caffe.NetSpec()

    conv_sigmoid3_n.data = L.DummyData(shape=dict(dim=[1, 3, 28, 28]))
    conv_sigmoid3_n.conv1 = L.Convolution(conv_sigmoid3_n.data, kernel_size=1, num_output=512, stride=1,
                                          weight_filler=dict(type='xavier'),
                                          bias_filler=dict(type='constant'))
    conv_sigmoid3_n.sigmoid1 = L.Sigmoid(conv_sigmoid3_n.conv1)

    conv_sigmoid3.setOutput('sigmoid1')

    # conv_sigmoid4
    conv_sigmoid4 = TopologyElement('conv_sigmoid4', ['conv', 'sigmoid'], None)
    conv_sigmoid4_path = path_origin + '/train_' + conv_sigmoid4.getName() + '.prototxt'
    conv_sigmoid4.setPath(conv_sigmoid4_path)
    conv_sigmoid4_n = caffe.NetSpec()

    conv_sigmoid4_n.data = L.DummyData(shape=dict(dim=[1, 3, 28, 28]))
    conv_sigmoid4_n.conv1 = L.Convolution(conv_sigmoid4_n.data, kernel_size=1, num_output=1024, stride=1,
                                          weight_filler=dict(type='xavier'),
                                          bias_filler=dict(type='constant'))
    conv_sigmoid4_n.sigmoid1 = L.Sigmoid(conv_sigmoid4_n.conv1)

    conv_sigmoid4.setOutput('sigmoid1')



    return (
        conv_pool1,
        conv_pool2,
        relu_softmax1_n,
        conv_relu_pool1,
        conv_relu_pool2,
        conv_relu_pool3,
        conv_relu_pool4,
        conv_sigmoid1,
        conv_sigmoid2,
        conv_sigmoid3,
        conv_sigmoid4
    )


topologies = generate_topologies()


def gen_train_proto_single(target_interface, shape=(28, 28, 3)):
    path = path_origin + '/train_' + target_interface + '.prototxt'
    n = caffe.NetSpec()

    n.data = L.DummyData(shape=dict(dim=[1, 3, shape[0], shape[1]]))
    if target_interface == 'conv1':
        n.conv1 = L.Convolution(n.data, kernel_size=11, stride=4,
                                num_output=1, weight_filler={"type": "constant", "value": 1})
    elif target_interface == 'pool1':
        n.pool1 = L.Pooling(n.data, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    elif target_interface == 'pool2':
        n.pool2 = L.Pooling(n.data, kernel_size=2, stride=2, pool=P.Pooling.AVE)
    elif target_interface == 'relu1':
        n.relu1 = L.ReLU(n.data)
    elif target_interface == 'sigmoid1':
        n.sigmoid1 = L.Sigmoid(n.data)
    elif target_interface == 'softmax1':
        n.softmax1 = L.Softmax(n.data)
    elif target_interface == 'tanh1':
        n.tanh1 = L.TanH(n.data)

    save_proto(n.to_proto(), path)
    return path


def gen_train_proto_multiple(target_interface, shape=(28, 28, 3)):
    path = path_origin + '/train_' + target_interface + 'm.prototxt'
    n = caffe.NetSpec()

    n.data = L.DummyData(shape=dict(dim=[1, 3, shape[0], shape[1]]))
    if target_interface == 'conv1':
        n.conv1_1 = L.Convolution(n.data, kernel_size=11, stride=4,
                                  num_output=1, weight_filler={"type": "constant", "value": 1})
        n.conv1_2 = L.Convolution(n.conv1_1, kernel_size=11, stride=4,
                                  num_output=1, weight_filler={"type": "constant", "value": 1})
        n.conv1_3 = L.Convolution(n.conv1_2, kernel_size=11, stride=4,
                                  num_output=1, weight_filler={"type": "constant", "value": 1})
        n.conv1_4 = L.Convolution(n.conv1_3, kernel_size=11, stride=4,
                                  num_output=1, weight_filler={"type": "constant", "value": 1})
        n.conv1_5 = L.Convolution(n.conv1_5, kernel_size=11, stride=4,
                                  num_output=1, weight_filler={"type": "constant", "value": 1})
    elif target_interface == 'pool1':
        n.pool1 = L.Pooling(n.data, kernel_size=2, stride=2, pool=P.Pooling.MAX)
        n.pool2 = L.Pooling(n.pool1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
        n.pool3 = L.Pooling(n.pool2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
        n.pool4 = L.Pooling(n.pool3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
        n.pool5 = L.Pooling(n.pool4, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    elif target_interface == 'pool2':
        n.pool2_1 = L.Pooling(n.data, kernel_size=2, stride=2, pool=P.Pooling.AVE)
        n.pool2_2 = L.Pooling(n.pool2_1, kernel_size=2, stride=2, pool=P.Pooling.AVE)
        n.pool2_3 = L.Pooling(n.pool2_2, kernel_size=2, stride=2, pool=P.Pooling.AVE)
        n.pool2_4 = L.Pooling(n.pool2_3, kernel_size=2, stride=2, pool=P.Pooling.AVE)
        n.pool2_5 = L.Pooling(n.pool2_4, kernel_size=2, stride=2, pool=P.Pooling.AVE)

    elif target_interface == 'relu1':
        n.relu1_1 = L.ReLU(n.data)
        n.relu1_2 = L.ReLU(n.relu1_1)
        n.relu1_3 = L.ReLU(n.relu1_2)
        n.relu1_4 = L.ReLU(n.relu1_3)
        n.relu1_5 = L.ReLU(n.relu1_4)

    elif target_interface == 'sigmoid1':
        n.sigmoid1_1 = L.Sigmoid(n.data)
        n.sigmoid1_2 = L.Sigmoid(n.sigmoid1_1)
        n.sigmoid1_3 = L.Sigmoid(n.sigmoid1_2)
        n.sigmoid1_4 = L.Sigmoid(n.sigmoid1_3)
        n.sigmoid1_5 = L.Sigmoid(n.sigmoid1_4)

    elif target_interface == 'softmax1':
        n.softmax1_1 = L.Softmax(n.data)
        n.softmax1_2 = L.Softmax(n.softmax1_1)
        n.softmax1_3 = L.Softmax(n.softmax1_2)
        n.softmax1_4 = L.Softmax(n.softmax1_3)
        n.softmax1_5 = L.Softmax(n.softmax1_4)

    elif target_interface == 'tanh1':
        n.tanh1_1 = L.TanH(n.data)
        n.tanh1_2 = L.TanH(n.tanh1_1)
        n.tanh1_3 = L.TanH(n.tanh1_2)
        n.tanh1_4 = L.TanH(n.tanh1_3)
        n.tanh1_5 = L.TanH(n.tanh1_4)

    save_proto(n.to_proto(), path)
    return path


def gen_solver_proto(path):
    from caffe.proto import caffe_pb2
    s = caffe_pb2.SolverParameter()

    # 为重现试验设置随机种子
    # 控制训练过程的随机
    s.random_seed = 0xCAFFE

    s.train_net = path  # 指定网络训练配置文件的位置
    s.test_net.append(path_origin + '/test.prototxt')  # 指定网络测试配置文件的位置，可以添加多个测试网络
    s.test_interval = 500  # 每训练500迭代，测试一次
    s.test_iter.append(100)  # 每次测试时都要测试100个batch取平均
    s.iter_size = 1  # 处理batch_size*iter_size个数据后，更新一次参数
    s.max_iter = 10000  # 最大迭代次数

    # 求解器类型包括 "SGD", "Adam", "Nesterov" 等
    s.type = "SGD"

    # 设置基础学习率
    s.base_lr = 0.01

    # 设置momentum来加速学习，通过将当前和之前的updates进行加权平均
    s.momentum = 0.9
    # 设置权重衰减系数来正则化避免过拟合
    s.weight_decay = 5e-4

    # 设置`lr_policy`来定义学习率在训练期间如何变化。

    s.lr_policy = 'inv'
    s.gamma = 0.0001
    s.power = 0.75

    # 保持学习率不变(与自适应方法相对立)
    # s.lr_policy = 'fixed'

    # 每迭代stepsize次，学习率乘以gamma
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # 每迭代1000次显示当前训练损失和精度
    s.display = 1000

    # 设置在CPU还是GPU上训练
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    save_proto(s, path_origin + '/solver_cpu.prototxt')
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    save_proto(s, path_origin + '/solver_gpu.prototxt')
    return path_origin + '/solver_cpu.prototxt', path_origin + '/solver_gpu.prototxt'


def caffe_compute(data, target_interface, solver_path_gpu, solver_path_cpu, GPU_mode=1):
    if GPU_mode == 2:
        # caffe.set_mode_cpu()
        solver_cpu = caffe.SGDSolver(solver_path_cpu)
        # 得到data的形状，这里的图片是默认matplotlib底层加载的
        transformer = caffe.io.Transformer({'data': solver_cpu.net.blobs['data'].data.shape})

        # 把channel放到前面
        transformer.set_transpose('data', (2, 0, 1))
        # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))

        # # 图片像素放大到[0-255]
        # transformer.set_raw_scale('data', 255)

        # RGB-->BGR转换
        transformer.set_channel_swap('data', (2, 1, 0))

        # 计算结果
        solver_cpu.net.blobs['data'].data[...] = transformer.preprocess('data', data)
        solver_cpu.step(1)
        cpu_out = np.rollaxis(solver_cpu.net.blobs[target_interface].data, 1, 4)
        cpu_input_data = np.rollaxis(solver_cpu.net.blobs['data'].data, 1, 4)
        return None, None, cpu_out, cpu_input_data

    cpu_out = None
    cpu_input_data = None
    caffe.set_device(0)
    solver_gpu = caffe.SGDSolver(solver_path_gpu)
    # 得到data的形状，这里的图片是默认matplotlib底层加载的
    transformer = caffe.io.Transformer({'data': solver_gpu.net.blobs['data'].data.shape})

    # 把channel放到前面
    transformer.set_transpose('data', (2, 0, 1))
    # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))

    # # 图片像素放大到[0-255]
    # transformer.set_raw_scale('data', 255)

    # RGB-->BGR转换
    transformer.set_channel_swap('data', (2, 1, 0))

    # 计算结果
    solver_gpu.net.blobs['data'].data[...] = transformer.preprocess('data', data)
    solver_gpu.step(1)
    out = np.rollaxis(solver_gpu.net.blobs[target_interface].data, 1, 4)
    input_data = np.rollaxis(solver_gpu.net.blobs['data'].data, 1, 4)
    if GPU_mode:
        caffe.set_mode_cpu()
        solver_cpu = caffe.SGDSolver(solver_path_cpu)
        # 得到data的形状，这里的图片是默认matplotlib底层加载的
        transformer = caffe.io.Transformer({'data': solver_cpu.net.blobs['data'].data.shape})

        # 把channel放到前面
        transformer.set_transpose('data', (2, 0, 1))
        # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))

        # # 图片像素放大到[0-255]
        # transformer.set_raw_scale('data', 255)

        # RGB-->BGR转换
        transformer.set_channel_swap('data', (2, 1, 0))

        # 计算结果
        solver_cpu.net.blobs['data'].data[...] = transformer.preprocess('data', data)
        solver_cpu.step(1)
        cpu_out = np.rollaxis(solver_cpu.net.blobs[target_interface].data, 1, 4)
        cpu_input_data = np.rollaxis(solver_cpu.net.blobs['data'].data, 1, 4)

    # 返回数据
    if GPU_mode == 1:
        return out, input_data, cpu_out, cpu_input_data
    else:
        return out, input_data, None, None


def caffe_compute_single(data, target_interface, GPU_mode=1):
    # 获取CPU/GPU 数据
    solver_path_gpu = path_origin + '/solver_gpu' + '.prototxt'
    solver_path_cpu = path_origin + '/solver_cpu' + '.prototxt'
    gen_train_proto_single(target_interface, data.shape)
    return caffe_compute(data, target_interface, solver_path_cpu, solver_path_gpu, GPU_mode)


def caffe_compute_multiple(data, target_interface, GPU_mode=1):
    # 获取CPU/GPU 数据
    solver_path_gpu = path_origin + '/solver_gpu' + '.prototxt'
    solver_path_cpu = path_origin + '/solver_cpu' + '.prototxt'
    target_interface = target_interface + '_5'

    return caffe_compute(data, target_interface, solver_path_cpu, solver_path_gpu, GPU_mode)


def caffe_compute_combination(data, target_interface, GPU_mode=1):
    # 此处待讨论
    outputs = []
    inputs = []
    outputs_cpu = []
    inputs_cpu = []
    topology_list = []
    for topology in topologies:
        if topology.match(target_interface):
            topology_list.append(topology)
            solver_path_cpu, solver_path_gpu = gen_solver_proto(topology.getPath())
            output_name = topology.getOutput()
            output_gpu, input_gpu, output_cpu, input_cpu = \
                caffe_compute(data, output_name, solver_path_cpu, solver_path_gpu, GPU_mode)
            outputs.append(output_gpu)
            inputs.append(input_gpu)
            outputs_cpu.append(output_cpu)
            inputs_cpu.append(input_cpu)

    return outputs, inputs, outputs_cpu, inputs_cpu, topology_list


if __name__ == '__main__':
    print()
