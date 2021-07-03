import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import datetime

__all__ = ['sparseAE']
mnist = input_data.read_data_sets("../MNIST_data/")
print(mnist.validation.images.shape)
x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


class DenoisingAutoEncoder(object):
    def __init__(self, m, n, eta=0.01):
        """
        :param m: 输入/输出层神经元的个数
        :param n: 隐含层神经与的个数
        :param eta:学习率
        """
        self._m = m
        self._n = n
        self.learning_rate = eta

        # 创建计算图

        # 权值和阈值
        self._W1 = tf.Variable(tf.random_normal(shape=(self._m, self._n)))
        self._W2 = tf.Variable(tf.random_normal(shape=(self._n, self._m)))
        # 隐藏层的阈值
        self._b1 = tf.Variable(np.zeros(self._n).astype(np.float32))
        # 输出层的阈值
        self._b2 = tf.Variable(np.zeros(self._m).astype(np.float32))

        # 输入占位符
        self._X = tf.placeholder('float', [None, self._m])
        self._X_noisy = tf.placeholder('float', [None, self._m])
        self.y = self.encoder(self._X_noisy)
        self.r = self.decoder(self.y)
        error = self._X - self.r

        self._loss = tf.reduce_mean(tf.pow(error, 2))
        alpha = 7.5e-5
        k1_div_loss = tf.reduce_sum(self.k1_div(0.02, tf.reduce_mean(self.y, 0)))
        loss = self._loss + alpha * k1_div_loss
        self._opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    # 编码函数
    def encoder(self, x):
        h = tf.matmul(x, self._W1) + self._b1
        return tf.nn.sigmoid(h)

    # 解码函数
    def decoder(self, x):
        h = tf.matmul(x, self._W2) + self._b2
        return tf.nn.sigmoid(h)

    def set_session(self, session):
        self.session = session

    def reduced_dimension(self, x):
        h = self.encoder(x)
        return self.session.run(h, feed_dict={self._X: x})

    def reconstruct(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return self.session.run(r, feed_dict={self._X: x})

    # 计算散度
    def k1_div(self, rho, rho_hat):
        term2_num = tf.constant(1.) - rho
        term2_den = tf.constant(1.) - rho_hat
        k1 = self.logfunc(rho, rho_hat) + self.logfunc(term2_num, term2_den)
        return k1

    def logfunc(self, x1, x2):
        return tf.multiply(x1, tf.log(tf.div(x1, x2)))

    def fit(self, X, epochs=1, batch_size=100):
        N, D = X.shape
        num_batches = N // batch_size
        obj = []
        noise_factor = 0.5
        for i in range(epochs):
            for j in range(num_batches):
                orgimgs = X[j * batch_size: (j * batch_size + batch_size)]
                noisy_imgs = orgimgs + noise_factor * np.random.randn(*orgimgs.shape)
                noisy_imgs = np.clip(noisy_imgs, 0., 1.)
                _, ob = self.session.run([self._opt, self._loss],
                                         feed_dict={self._X: orgimgs, self._X_noisy: noisy_imgs})
                if j % 100 == 0 and i % 100 == 0:
                    print('Training epoch{0} batch {2} cost {1}'.format(i, ob, j))
                obj.append(ob)
        return obj


def sparseAE(adv_example):
    # 定义计算图并新开一个session,避免和原有的session冲突
    sparseAE_graph = tf.Graph()
    sess1 = tf.Session(graph=sparseAE_graph)
    with sess1.as_default():
        with sparseAE_graph.as_default():
            # 隐藏神经元个数
            n_hidden = 800
            Xtrain = x_train.astype(np.float32)
            _, m = Xtrain.shape
            print(m)
            autoEncoder = DenoisingAutoEncoder(m, n_hidden)
            ckpt_state = tf.train.get_checkpoint_state('./model/denoisesparseAE_model/')
            if ckpt_state:
                saver = tf.train.Saver()
                saver.restore(sess1, ckpt_state.model_checkpoint_path)
                autoEncoder.set_session(sess1)
                # 保存去噪后的图片
                # 将图像分成100组数据，每组100个数据（myimage的shape为（10000,28,28,1））
                fgsm_image = adv_example.reshape((100, 100, 28, 28, 1))
                for i in range(100):
                    fgsm_image[i] = autoEncoder.reconstruct(fgsm_image[i].reshape(100,784)).reshape(100,28,28,1)

            else:
                init = tf.global_variables_initializer()
                sess1.run(init)
                autoEncoder.set_session(sess1)
                err = autoEncoder.fit(Xtrain, epochs=10)
                # 保存模型
                saver = tf.train.Saver()
                saver.save(sess1, './model/denoisesparseAE_model/model')
                # 保存去噪后的图片
                # 将图像分成100组数据，每组100个数据（myimage的shape为（10000,28,28,1））
                fgsm_image = adv_example.reshape((100, 100, 28, 28, 1))
                for i in range(100):
                    fgsm_image[i] = autoEncoder.reconstruct(fgsm_image[i])
    return fgsm_image.reshape((10000, 28, 28, 1))
