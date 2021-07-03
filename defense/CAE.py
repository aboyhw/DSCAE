import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


__all__ = ['CAE']

mnist = input_data.read_data_sets("../MNIST_data/")
print(mnist.validation.images.shape)
x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


def CAE(adv_example):
    # 定义计算图并新开一个session,避免和原有的session冲突
    CAE_graph = tf.Graph()
    sess1 = tf.Session(graph=CAE_graph)
    with sess1.as_default():
        with CAE_graph.as_default():
            # g1 = tf.Graph()
            # 网络参数
            # 图像的高度和宽度
            h_in, w_in = 28, 28
            # 卷积核的大小
            k = 3
            # 池化
            p = 2
            # 步长
            s = 2
            # 过滤器深度(以字典的形式)
            filters = {1: 32, 2: 32, 3: 16}
            activation_fn = tf.nn.relu
            # np.ceil()计算大于等于该值的最小整数（向上取整），因为后面采用了padding='same'全零填充
            # 卷积后的矩阵大小计算公式
            h_l2, w_l2 = int(np.ceil(float(h_in) / float(s))), int(np.ceil(float(w_in) / float(s)))
            h_l3, w_l3 = int(np.ceil(float(h_l2) / float(s))), int(np.ceil(float(w_l2) / float(s)))

            # 为输入（噪声）和目标（对应清晰图像）创建占位符
            X_noisy = tf.placeholder(tf.float32, (None, h_in, w_in, 1), name='inputs')
            X = tf.placeholder(tf.float32, (None, h_in, w_in, 1), name='targets')

            # Encoder
            # 默认步长为（1,1）
            conv1 = tf.layers.conv2d(X_noisy, filters[1], (k, k), padding='same', activation=activation_fn)
            # 经过conv1后，输出大小为 h_in * w_in * filters[1]
            maxpool1 = tf.layers.max_pooling2d(conv1, (p, p), (s, s), padding='same')
            # 经过maxpool1后，输出大小为 h_l2 * w_l2 * filters[1]
            conv2 = tf.layers.conv2d(maxpool1, filters[2], (k, k), padding='same', activation=activation_fn)
            # 经过conv2后，输出大小为 h_l2 * w_l2 * filters[2]
            maxpool2 = tf.layers.max_pooling2d(conv2, (p, p), (s, s), padding='same')
            # 经过maxpool2后，输出大小为 h_l3 * w_l3 * filters[2]
            conv3 = tf.layers.conv2d(maxpool2, filters[3], (k, k), padding='same', activation=activation_fn)
            # 经过conv3后，输出大小为 h_l3 * w_l3 * filters[3]
            encoded = tf.layers.max_pooling2d(conv3, (p, p), (s, s), padding='same')
            # 最终输出大小为 (h_l3/s) * (w_l3/s) * filters[3]

            # Decoder
            upsample1 = tf.image.resize_nearest_neighbor(encoded, (h_l3, w_l3))
            # 经过upsample1后，输出大小为 h_l3 * w_l3 * filters[3]
            conv4 = tf.layers.conv2d(upsample1, filters[3], (k, k), padding='same', activation=activation_fn)
            # 经过conv4后，输出大小为 h_l3 * w_l3 * filters[3]
            upsample2 = tf.image.resize_nearest_neighbor(conv4, (h_l2, w_l2))
            # 经过upsample2 后，输出大小为 h_l2 * w_l2 * filters[3]
            conv5 = tf.layers.conv2d(upsample2, filters[2], (k, k), padding='same', activation=activation_fn)
            # 经过conv5 后，输出大小为 h_l2 * w_l2 * filters[2]
            upsample3 = tf.image.resize_nearest_neighbor(conv5, (h_in, w_in))
            # 经过upsample3后，输出大小为 h_in * w_in * filters[2]
            conv6 = tf.layers.conv2d(upsample3, filters[1], (k, k), padding='same', activation=activation_fn)
            # 最终输出大小为 (h_in) * (w_in) * filters[1]
            logits = tf.layers.conv2d(conv6, 1, (k, k), padding='same', activation=None)
            # 输出 h_in * w_in *1 shape的图像
            decoded = tf.nn.sigmoid(logits, name='decoded')

            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
            cost = tf.reduce_mean(loss)
            opt = tf.train.AdamOptimizer(0.001).minimize(cost)
            tf.image.draw_bounding_boxes

            ckpt_state = tf.train.get_checkpoint_state('./model/CAE_model/')
            if ckpt_state:
                saver = tf.train.Saver()
                saver.restore(sess1, ckpt_state.model_checkpoint_path)
            else:
                epochs = 10
                batch_size = 100
                noise_factor = 0.5
                sess1.run(tf.global_variables_initializer())
                err = []
                for i in range(epochs):
                    for ii in range(mnist.train.num_examples // batch_size):
                        batch = mnist.train.next_batch(batch_size)
                        # reshape((-1, h_in, w_in, 1))取出图像，-1表示取所有的行
                        imgs = batch[0].reshape((-1, h_in, w_in, 1))
                        noisy_imgs = imgs + noise_factor * np.random.randn(* imgs.shape)
                        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
                        batch_cost, _ = sess1.run([cost, opt], feed_dict={X_noisy: noisy_imgs, X: imgs})
                        err.append(batch_cost)
                        if ii % 100 == 0:
                            print("Epoch: {0}/{1}...Training loss {2}".format(i, epochs, batch_cost))

                # 保存模型
                saver = tf.train.Saver()
                saver.save(sess1, './model/CAE_model/model')

            # 保存去噪后的图片
            # 将图像分成100组数据，每组100个数据（myimage的shape为（10000,28,28,1））
            fgsm_image = adv_example.reshape((100,100,28,28,1))
            for i in range(100):
                fgsm_image[i] = sess1.run(decoded, feed_dict={X_noisy: fgsm_image[i]})
            # 关闭会话

    return fgsm_image.reshape((10000,28,28,1))
