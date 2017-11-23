# coding: utf-8

import tensorflow as tf

matrix1 = tf.Variable([[3., 3.]], name="variable_1")
matrix2 = tf.Variable([[2.], [2.]], name="variable_2")

init = tf.initialize_all_variables()
saver = tf.train.Saver()

product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
sess.run(init)

result = sess.run(product)
print(sess.run(matrix1))
print(sess.run(matrix2))
print(result)
saver.save(sess, "model/sample/test.ckpt")

sess.close()
