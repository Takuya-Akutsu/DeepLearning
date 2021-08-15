import numpy as np
import matplotlib.pyplot as plt
import mnist as mn
from deep_convnet import DeepConvNet
from trainer import Trainer

(x_train, t_train), (x_test, t_test) = mn.load_mnist(flatten=False)

network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")