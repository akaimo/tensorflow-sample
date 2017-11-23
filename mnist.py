# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])         # 入力される画像データ
W = tf.Variable(tf.zeros([784,10]), name="weight")  # 重み
b = tf.Variable(tf.zeros([10]), name="bias")        # バイアス

y = tf.nn.softmax(tf.matmul(x, W) + b)              # NNの出力
y_ = tf.placeholder(tf.float32, [None, 10])         # 教師データなどの正解

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

# logを保存
tf.summary.scalar('cross_entropy', cross_entropy)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("log", sess.graph)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    feed_dict = {x: batch_xs, y_: batch_ys}
    sess.run(train_step, feed_dict=feed_dict)
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, i)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver.save(sess, "model/mnist/mnist.ckpt")

print(sess.run(b))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
