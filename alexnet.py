import mxnet as mx
from mxnet import autograd, nd
import mxnet.gluon.nn as nn
from mxnet import init
import os
from mxnet.gluon import data as gdata, utils as gutils
from time import time
from mxnet import gluon
import mxnet.gluon.loss as gloss
import gluonbook as gb
import sys

net = nn.Sequential()
# 224*224*3
# 减半模块
net.add(
    # 11*11捕获物体，减小输出高宽
    nn.Conv2D(96, kernel_size=11, strides=4, activation="relu"),
    # 减小卷积窗口，使输出高宽一致
    nn.MaxPool2D(pool_size=3, strides=2),
    # 27*27*96
    nn.BatchNorm(),
    nn.Conv2D(256, kernel_size=5, padding=2, activation="relu"),
    nn.MaxPool2D(pool_size=3, strides=2),
    nn.BatchNorm(),
    # 13*13*256
    nn.Conv2D(384, padding=1, kernel_size=3, activation="relu"),
    nn.Conv2D(384, padding=1, kernel_size=3, activation="relu"),
    nn.Conv2D(256, kernel_size=3, padding=1, activation="relu"),
    nn.MaxPool2D(pool_size=3, strides=2),
    # fc 1
    nn.Dense(4096, activation="relu"),
    nn.Dropout(.5),
    nn.Dense(4096, activation="relu"),
    nn.Dropout(0.5),
    nn.Dense(10)
)
net.initialize()
x = nd.random_uniform(shape=[1, 1, 224, 224])
print(net(x))


def load_data_fashion_mnist(batch_size, resize=None,
                            root=os.path.join('~', '.mxnet', 'datasets',
                                              'fashion-mnist')):
    root = os.path.expanduser(root)  # 展开用户路径 '~'。
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle=True,
        num_workers=num_workers)
    test_iter = gdata.DataLoader(
        mnist_test.transform_first(transformer), batch_size, shuffle=False,
        num_workers=num_workers)
    return train_iter, test_iter


batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size,
                                                resize=224)
lr = 0.01
num_epochs = 5
ctx = gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
gb.train_ch5(net, train_iter, test_iter, loss, batch_size, trainer, ctx,
             num_epochs)
