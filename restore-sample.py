# coding: utf-8

import tensorflow as tf

matrix1 = tf.Variable([[0., 0.]], name="variable_1")
matrix2 = tf.Variable([[0.], [0.]], name="variable_2")
product = tf.matmul(matrix1, matrix2)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "model/sample/test.ckpt")

result = sess.run(product)
print(sess.run(matrix1))
print(sess.run(matrix2))
print(result)

sess.close()
