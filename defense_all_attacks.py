import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from attacks import *
from defense import *
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from tensorflow.keras import datasets, layers, models
from tensorflow.examples.tutorials.mnist import input_data
from timeit import default_timer


img_size = 28
img_chan = 1
n_classes = 10
batch_size = 32


class Timer(object):
    def __init__(self, msg='Starting.....', timer=default_timer, factor=1,
                 fmt="------- elapsed {:.4f}s --------"):
        self.timer = timer
        self.factor = factor
        self.fmt = fmt
        self.end = None
        self.msg = msg

    def __call__(self):
        """
        Return the current time
        """
        return self.timer()

    def __enter__(self):
        """
        Set the start time
        """
        print(self.msg)
        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Set the end time
        """
        self.end = self()
        print(str(self))

    def __repr__(self):
        return self.fmt.format(self.elapsed)

    @property
    def elapsed(self):
        if self.end is None:
            # if elapsed is called in the context manager scope
            return (self() - self.start) * self.factor
        else:
            # if elapsed is called out of the context manager scope
            return (self.end - self.start) * self.factor


print('\nloading MNIST')
mnist = input_data.read_data_sets("./MNIST_data/")

# mnist = tf.keras.datasets.mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train, X_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
'''
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
'''
X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
#X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
#X_test = X_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\nspliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]

print('\nConstruction graph')


# 定义模型
def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)

    env.saver = tf.train.Saver()


with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    # fgsm
    env.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
    env.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
    env.x_fgmt = fgmt(model, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)
    # deepfool
    env.adv_epochs_deepfool = tf.placeholder(tf.int32, (), name='adv_epochs')
    env.xadv_deepfool = deepfool(model, env.x, epochs=env.adv_epochs_deepfool)
    # jsma
    env.target = tf.placeholder(tf.int32, (), name='target')
    env.adv_epochs = tf.placeholder_with_default(20, shape=(), name='epochs')
    env.adv_eps = tf.placeholder_with_default(0.2, shape=(), name='eps')
    env.x_jsma = jsma(model, env.x, env.target, eps=env.adv_eps,
                      epochs=env.adv_epochs)

    # cw
    env.x_fixed = tf.placeholder(
        tf.float32, (batch_size, img_size, img_size, img_chan),
        name='x_fixed')
    env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
    env.adv_y = tf.placeholder(tf.int32, (), name='adv_y')

    # optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    env.adv_train_op, env.xadv, env.noise = cw(model, env.x_fixed,
                                               y=env.adv_y, eps=env.adv_eps)

print('\niniting graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def evaluate(sess, env, X_data, y_data, batch_size=100):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating......')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=100, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTraining mocel')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nepochs {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(sess, 'model/{}'.format(name))


def predict(sess, env, X_data, batch_size=100):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting......')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch{0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_fgmt(sess, env, X_data, epochs=10, eps=0.01, batch_size=100):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print('\nGenerate FGSM adversarial examples')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv_fgsm = np.empty_like(X_data)
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgmt, feed_dict={
            env.x: X_data[start:end],
            env.fgsm_eps: eps,
            env.fgsm_epochs: epochs})
        X_adv_fgsm[start:end] = adv
    print()

    return X_adv_fgsm


def make_deepfool(sess, env, X_data, epochs=1, eps=0.01, batch_size=100):
    """
    Generate DeepFool by running env.xadv.
    """
    print('\nMaking adversarials via DeepFool')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv_deepfool = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.xadv_deepfool, feed_dict={env.x: X_data[start:end],
                                            env.adv_epochs_deepfool: epochs})
        X_adv_deepfool[start:end] = adv
    print()

    return X_adv_deepfool


def make_jsma(sess, env, X_data, epochs=0.2, eps=1.0, batch_size=100):
    """
    Generate JSMA by running env.x_jsma.
    """
    print('\nMaking adversarials via JSMA')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv_jsma = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {
            env.x: X_data[start:end],
            env.target: np.random.choice(n_classes),
            env.adv_epochs: epochs,
            env.adv_eps: eps}
        adv = sess.run(env.x_jsma, feed_dict=feed_dict)
        X_adv_jsma[start:end] = adv
    print()

    return X_adv_jsma


def make_cw(sess, env, X_data, epochs=1, eps=0.1, batch_size=batch_size):
    """
    Generate adversarial via CW optimization.
    """
    print('\nMaking adversarials via CW')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        with Timer('Batch {0}/{1}   '.format(batch + 1, n_batch)):
            end = min(n_sample, (batch+1) * batch_size)
            start = end - batch_size
            feed_dict = {
                env.x_fixed: X_data[start:end],
                env.adv_eps: eps,
                env.adv_y: np.random.choice(n_classes)}

            # reset the noise before every iteration
            sess.run(env.noise.initializer)
            for epoch in range(epochs):
                sess.run(env.adv_train_op, feed_dict=feed_dict)

            xadv = sess.run(env.xadv, feed_dict=feed_dict)
            X_adv[start:end] = xadv

    return X_adv


print('\nTraining')

train(sess, env, X_train, y_train, X_valid, y_valid, load=False, epochs=20,
      name='mnist')

print('\nEvaluating on clean data')

evaluate(sess, env, X_test, y_test)

# FGSM
print('\nFGSM:Generating  fgsm adversarial data')

X_adv1 = make_fgmt(sess, env, X_test, eps=0.02, epochs=12, batch_size=128)

print('\nFGSM:Evaluating on  fgsm adversarial data')

evaluate(sess, env, X_adv1, y_test)

print('\nFGSM:Generating sparseAE clean data')
ae_adv = X_adv1.copy()
ae_image = sparseAE(ae_adv)
print('\nFGSM:Evaluating on sparseAE clean data')
evaluate(sess, env, ae_image, y_test)
print('\nFGSM:Generating CAE clean data')
cae_adv = X_adv1.copy()
cae_image = CAE(cae_adv)
print('\nFGSM:Evaluating on CAE clean data')
evaluate(sess, env, cae_image, y_test)

print('\nFGSM:Generating sparse CAE clean data')
my_adv = X_adv1.copy()
sparse_cae_image = spcae(my_adv)
print('\nFGSM:Evaluating on sparse CAE clean data')

evaluate(sess, env, sparse_cae_image, y_test)
print('\n ............................................................')

# JSMA
X_adv2 = make_jsma(sess, env, X_test, epochs=30, eps=1.0)

print('\nJSMA:Evaluating on  jsma adversarial data')

evaluate(sess, env, X_adv2, y_test)

print('\nJSMA:Generating sparseAE clean data')
ae_adv = X_adv2.copy()
ae_image = sparseAE(ae_adv)
print('\nJSMA:Evaluating on sparseAE clean data')
evaluate(sess, env, ae_image, y_test)
print('\nJSMA:Generating CAE clean data')
cae_adv = X_adv2.copy()
cae_image = CAE(cae_adv)
print('\nJSMA:Evaluating on CAE clean data')
evaluate(sess, env, cae_image, y_test)

print('\nJSMA:Generating sparse CAE clean data')
my_adv = X_adv2.copy()
sparse_cae_image = spcae(my_adv)
print('\nJSMA:Evaluating on sparse CAE clean data')

evaluate(sess, env, sparse_cae_image, y_test)

print('\n ............................................................')

# DEEPFOOL
X_adv3 = make_deepfool(sess, env, X_test, epochs=10)
print('\nDEEPFOOL:Evaluating on  fgsm adversarial data')

evaluate(sess, env, X_adv3, y_test)

print('\nDEEPFOOL:Generating sparseAE clean data')
ae_adv = X_adv3.copy()
ae_image = sparseAE(ae_adv)
print('\nDEEPFOOL:Evaluating on sparseAE clean data')
evaluate(sess, env, ae_image, y_test)
print('\nDEEPFOOL:Generating CAE clean data')
cae_adv = X_adv3.copy()
cae_image = CAE(cae_adv)
print('\nDEEPFOOL:Evaluating on CAE clean data')
evaluate(sess, env, cae_image, y_test)

print('\nDEEPFOOL:Generating sparse CAE clean data')
my_adv = X_adv3.copy()
sparse_cae_image = spcae(my_adv)
print('\nDEEPFOOL:Evaluating on sparse CAE clean data')

evaluate(sess, env, sparse_cae_image, y_test)

print('\n ............................................................')

#cw
X_adv4 = make_cw(sess, env, X_test, eps=0.002, epochs=100)
print('\nCW:Evaluating on  fgsm adversarial data')

evaluate(sess, env, X_adv4, y_test)

print('\nCW:Generating sparseAE clean data')
ae_adv = X_adv4.copy()
ae_image = sparseAE(ae_adv)
print('\nCW:Evaluating on sparseAE clean data')
evaluate(sess, env, ae_image, y_test)
print('\nCW:Generating CAE clean data')
cae_adv = X_adv4.copy()
cae_image = CAE(cae_adv)
print('\nCW:Evaluating on CAE clean data')
evaluate(sess, env, cae_image, y_test)

print('\nCW:Generating sparse CAE clean data')
my_adv = X_adv4.copy()
sparse_cae_image = spcae(my_adv)
print('\nCW:Evaluating on sparse CAE clean data')

evaluate(sess, env, sparse_cae_image, y_test)
