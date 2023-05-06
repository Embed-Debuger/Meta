import os

os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np

# 设置网络结构
# net_file = 'E:/PycharmProjects/test-caffe/net/deploy.prototxt'
# path1 = os.path.abspath('..')
# net_file = path1 + '\\caffe-fuzz\\net\deploy.prototxt'
# print(net_file)
#
#
# # 添加训练之后的参数
# # caffe_model = 'E:/PycharmProjects/test-caffe/net/bvlc_reference_caffenet.caffemodel'
# caffe_model = path1 + '\\caffe-fuzz\\net\\bvlc_reference_caffenet.caffemodel'
#
# # 均值文件
# # mean_file = 'E:/PycharmProjects/test-caffe/net/ilsvrc_2012_mean.npy'
# mean_file = path1 + '\\caffe-fuzz\\net\ilsvrc_2012_mean.npy'
#
# # imagenet_labels_filename = 'E:/PycharmProjects/test-caffe/net/synset_words.txt'
# imagenet_labels_filename = path1 + '\\caffe-fuzz\\net\synset_words.txt'
path_origin = './net'


def caffe_compute_all(im, target_interface, GPU_mode=1, comparFramework=''):
    # 获取CPU/GPU 数据
    solver_path_gpu = path_origin + '/lsolver_gpu_' + target_interface + '.prototxt'
    solver_path_cpu = path_origin + '/lsolver_cpu_' + target_interface + '.prototxt'
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
        solver_cpu.net.blobs['data'].data[...] = transformer.preprocess('data', im)
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
    solver_gpu.net.blobs['data'].data[...] = transformer.preprocess('data', im)
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
        solver_cpu.net.blobs['data'].data[...] = transformer.preprocess('data', im)
        solver_cpu.step(1)
        cpu_out = np.rollaxis(solver_cpu.net.blobs[target_interface].data, 1, 4)
        cpu_input_data = np.rollaxis(solver_cpu.net.blobs['data'].data, 1, 4)

    # 返回数据
    if GPU_mode == 1:
        return out, input_data, cpu_out, cpu_input_data
    else:
        return out, input_data, None, None


if __name__ == '__main__':
    im = caffe.io.load_image('D:/python workspace/test-caffe/corpus/cat.jpg')
    # print(im)
    caffe_compute_all(im)
