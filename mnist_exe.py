# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])         # 入力される画像データ
W = tf.Variable(tf.zeros([784,10]), name="weight")  # 重み
b = tf.Variable(tf.zeros([10]), name="bias")        # バイアス

y = tf.nn.softmax(tf.matmul(x, W) + b)              # NNの出力
y_ = tf.placeholder(tf.float32, [None, 10])         # 教師データなどの正解

init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

saver.restore(sess, "model/mnist/mnist.ckpt")

# print(sess.run(W))
# print(sess.run(b))

image, answer = mnist.train.next_batch(20)
print(sess.run(tf.argmax(y,1), feed_dict={x: image}))
print(sess.run(tf.argmax(answer, 1)))
