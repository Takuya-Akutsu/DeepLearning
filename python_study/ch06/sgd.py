# import twoLayerNet as tln

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

        # network = tln.TwoLayerNet(...)

# #optimizer:最適化を行う者
# optimizer = SGD()

# for i in range(10000) :
#         ...
#     x_batch, t_batch = get_mini_batch(...) #こんなの作ってないよね？
#     grads = network.gradient(x_batch, t_batch)
#     params = network.params
#     optimizer.update(params, grads)
#     ...