from test_fuzzer import *

if __name__=='__main__':

    Fuzz(itf='relu1', mode=2, DLFW='mxnet', DLFW_O='pytorch',test_style=0)
    Fuzz(itf='relu1', mode=2, DLFW='mxnet', DLFW_O='tensorflow', test_style=0)
    Fuzz(itf='relu1', mode=2, DLFW='tensorflow', DLFW_O='pytorch', test_style=0)
    Fuzz(itf='relu1', mode=2, DLFW='mnn', DLFW_O='tensorflow',test_style=0)
    Fuzz(itf='relu1', mode=2, DLFW='mnn', DLFW_O='pytorch', test_style=0)

