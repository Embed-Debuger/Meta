import caffe

solver = caffe.SGDSolver('mnist/lenet_solver.prototxt')
solver.solve()