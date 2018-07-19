# mnist
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf


mnist = input_data.read_data_sets("", one_hot=True)
# mnist.train.images 训练数据集图片
# mnist.train.labels 训练数据集标签
images = mnist.train.images     # 28*28*60000
labels = mnist.train.labels     # 10*60000
print(images.shape)

x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)

# train models
y_ = tf.placeholder("float", [None, 10])
cross_entorpy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entorpy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
sess.close()

