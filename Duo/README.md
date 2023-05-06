# dlfuzz

## Update4
- 存疑：是否可以不记录output值，从而节约内存消耗
- 增加了新的评估函数，用于测试对象对应多测试实体的情况，但未完全完成
- 完善了Caffe/Tensorflow下对于算子拓扑组合的支持
- 请吕军修改
    - 在算子拓扑情况下的实验数据记录方式
    - 我在tensorflow_compute中注释的代码有问题，但我完全不懂为什么，请吕军你看一下
        - Line:277,280
    - 完成新的针对拓扑结构的差分测试策略
    

## Update3
- 将mcmc/mcmc_flag参数调整为generate_strategy参数
- 调整了当coverage_function为None时的不合理种子output初始化问题
- 添加了组合接口测试模块，包括：
    - Fuzzer输入参数：test_style，用于区分单一接口测试和接口组合测试
    - 调整了Fuzzer类中的框架执行入口的接口名，使其和算子组合测试接口名对应
    - caffe调整
        - 修正caffe_compute.py的文件命名错误
        - 重构caffe_compute.py，将主要功能代码移动至caffe_compute()方法中，其余方法完成相关配置文件定位
        - 重构caffe_compute.py，自动化生成配置文件
        - 在Fuzzer类初始化阶段的decide_DLcomputer方法调用中，完成配置文件初始化
        - 在test_one_input过程中，根据input的shape调整配置文件输入层信息
        - 将caffe的执行区分为caffe_compute_single和caffe_compute_multiple两个接口，并最终调用caffe_compute这一具体运算接口
        - caffe_compute_combination未完全实现
    - tensorflow调整
        - 重构了tensorflow_compute.py，缩短了代码行数
        - 添加了tensorflow_compute_single和tensorflow_compute_multiple方法，支持两种运算
        

- 请吕军完成的任务
    - 请调整算子组合后的参数，包括kernel_size，strides等
    - 讨论caffe_compute_combination的返回方式
    - 添加pytorch的相应模块
      
## Update2
- 更新了复现代码：recurrent.py
- 需要吕军补充的任务
    - 目前我没有对应input数据，请吕军补充完recurrent.py代码
    - 请吕军对几批运行数据进行复现，并给我一份分析文档，说明出现较大差异的原因，包括：
        - 是否是因为输入中出现异常值
        - 是否是因为输入正常但算子实现存在差异
        - 是否是因为实验细节存在问题
    - 目前input的规模有问题，我猜测应该是14\*14\*3,但目前的结果是13\*14\*3,请吕军核查一下实验结果生成代码

## Update1
- 添加了扰动方法
    - 增加了4种插入值策略
    - 增加了两种插入值方法
    - 方法细节需要讨论
- 调整了评估策略
    - 增加了tol策略
    - 将原先的部分max评估修改为三合一评估
- 需要补充的任务
    - difference评估方法的调整
    - 数据记录方式调整
    - 初始化参数设定（目前还没做，评估应该跑不起来）
