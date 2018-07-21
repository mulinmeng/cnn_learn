import mxnet as mx
import mxnet.gluon.nn as nn
import gzip
import struct
import numpy as np
from mxnet.gluon import data as gdata
from mxnet import gluon
from mxnet.gluon import loss as gloss, utils as gutils
from time import time
from mxnet import autograd
from mxnet import nd
from mxnet import init

net = nn.Sequential()
net.add(
    nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    # Dense 会默认将（批量大小，通道，高，宽）形状的输入转换成
    # （批量大小，通道 x 高 x 宽）形状的输入。
    nn.Dense(120, activation='sigmoid'),
    nn.Dense(84, activation='sigmoid'),
    nn.Dense(10)
)
net.initialize()


def read_data(label_url,image_url):
    with gzip.open(label_url) as flbl:
        magic, num = struct.unpack(">II",flbl.read(8))
        # 采用Big Endian的方式读取两个int类型的数据，且参考MNIST官方格式介绍，magic即为magic number (MSB first) 用于表示文件格式
        # num即为文件夹内包含的数据的数量
        label = np.frombuffer(flbl.read(), dtype=np.int8)
        # 将标签包中的每一个二进制数据转化成其对应的十进制数据
        # 且转换后的数据格式为int8（-128 to 127）格式，返回一个数组
    with gzip.open(image_url,'rb') as fimg:
        # 已只读形式解压图像包
        magic, num, rows, cols = struct.unpack(">IIII",fimg.read(16))
        # 采用Big Endian的方式读取四个int类型数据，且参考MNIST官方格式介绍
        # magic和num上同，rows和cols即表示图片的行数和列数
        image = np.frombuffer(fimg.read(),dtype=np.uint8).reshape(len(label),rows,cols)
        # 将图片包中的二进制数据读取后转换成无符号的int8格式的数组，并且以标签总个数，行数，列数重塑成一个新的多维数组
    print("read data success")
    return label, image
    # 返回读取成功的label数组和image数组
    # 且fileobject.read(size)的时候是按照流的方式读取（可test）


test_labels, test_images = read_data("t10k-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz")
print("test labels shape", test_labels.shape)
print("test images shape", test_images.shape)
train_labels, train_images = read_data("train-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz")
print("train labels shape", train_labels.shape)
print("train images shape", train_images.shape)


def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255


def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    if isinstance(batch, mx.io.DataBatch):
        features = batch.data[0]
        labels = batch.label[0]
    else:
        features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx),
            features.shape[0])


def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0
    if isinstance(data_iter, mx.io.MXDataIter):
        data_iter.reset()
    for batch in data_iter:
        features, labels, batch_size = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc += (net(X).argmax(axis=1)==y).sum().copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()
    return acc.asscalar() / n


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs,
          print_batches=None):
    """Train and evaluate a model."""
    print('training on', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0.0, 0.0
        start = time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in ys])
            if print_batches and (i + 1) % print_batches == 0:
                print('batch %d, loss %f, train acc %f' % (
                    n, train_l_sum / n, train_acc_sum / m
                ))
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch, train_l_sum / n, train_acc_sum / m, test_acc,
                 time() - start))


train_images = nd.array(to4d(train_images))
test_images = nd.array(to4d(test_images))
train_labels = nd.array(train_labels)
test_labels = nd.array(test_labels)


batch_size = 100
net.initialize(force_reinit=True, ctx=mx.cpu(), init=init.Xavier())
dataset_tr = gdata.ArrayDataset(train_images, train_labels)
train_iter = gdata.DataLoader(dataset_tr, batch_size, shuffle=True)
dataset_te = gdata.ArrayDataset(test_images, test_labels)
test_iter = gdata.DataLoader(dataset_tr, batch_size, shuffle=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 1})
loss = gloss.SoftmaxCrossEntropyLoss()
num_epochs = 5
train(train_iter, test_iter, net, loss, trainer, mx.cpu(),num_epochs)
'''
for epoch in range(1, num_epochs + 1):
    for X, y in train_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    print(net(test_images).argmax(axis=1))
    print('epoch %d, loss: %f'
          % (epoch, loss(net(test_images), test_labels).mean().asnumpy()))
'''

