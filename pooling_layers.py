import tensorflow as tf
import math
import numpy as np
import random
from tqdm import tqdm


def test_net(inputs):
    with tf.variable_scope('test'):
        x = tf.layers.conv2d(inputs, 64, [3, 3], 2, 'same', activation=tf.nn.relu)

        # test fot overlapping pooling
        # x = overlapping_pooling(x, [3, 3], (2, 2), 'same', 'max pooling')

        # test for mixed pooling
        # x= mixed_pooling('add')(x,[3, 3], (2, 2), 'same',0.5,0.5)

        # test for stochastic_pooling
        # x=stochastic_pooling(x,(2,2),(2,2),'SAME',True)

        # test for SPP
        x = spatial_pyramid_pool(x)
    return x


def overlapping_pooling(input, window_size, strides, padding, mode):
    '''
    This is the function of overlapping pooling
    :param input: input tensor with shape [batch size, height, width, channels]
    :param window_size: the size of pooling window, array like. samples:[3,3]
    :param strides: tuple with integers. sample:(2,2)
    :param padding: padding mode,valid of same
    :param mode: max pooling or average pooling
    :return:pooled tensor
    '''
    if mode == 'max pooling':
        output = tf.layers.max_pooling2d(inputs=input, pool_size=window_size, strides=strides, padding=padding)
    elif mode == 'average pooling':
        output = tf.layers.average_pooling2d(inputs=input, pool_size=window_size, strides=strides, padding=padding)
    else:
        raise ValueError('undefined mode type!')
    return output


def mixed_pooling(mode):
    '''
    The implementation of Mixed pooling
    :param mode: 'add' or 'concatenate'
    :return: pooled tensor
    '''

    def add_avg_max_pool2d(inputs, window_size, strides, padding, scale1, scale2):
        '''
        mixed pooling using 'add' mode
        :param inputs: input tensor with shape [batch size, height, width, channels]
        :param window_size:  the size of pooling window, array like. samples:[3,3]
        :param strides: tuple with integers. sample:(2,2)
        :param padding: padding mode,valid of same
        :param scale1: weight for avg pooled features
        :param scale2: weight for max pooled features
        :return: pooled tensor
        '''
        x_avg = tf.layers.average_pooling2d(inputs=inputs, pool_size=window_size, strides=strides, padding=padding)
        x_max = tf.layers.max_pooling2d(inputs=inputs, pool_size=window_size, strides=strides, padding=padding)
        return scale1 * x_avg + scale2 * x_max

    def cat_avg_max_pool2d(inputs, window_size, strides, padding):
        '''
        mixed pooling using 'concatenate' mode
        '''
        x_avg = tf.layers.average_pooling2d(inputs=inputs, pool_size=window_size, strides=strides, padding=padding)
        x_max = tf.layers.max_pooling2d(inputs=inputs, pool_size=window_size, strides=strides, padding=padding)
        return tf.concat([x_avg, x_max], -1)

    if mode == 'add':
        return add_avg_max_pool2d

    elif mode == 'concatenate':
        return cat_avg_max_pool2d

    else:
        raise ValueError("You must choose the mode from 'add' or 'concatenate'!")


def spatial_pyramid_pool(inputs, levels=[1, 2, 4], name='SPP_layer', pool_type='max_pool'):
    '''
    SPP layer. Output is a tensor with shape [batch size,16*256+4*256+256]
    :param levels: list. blocks per row and column
    :param pool_type: 'max_pool' or 'avg_pool' used for each branch
    '''

    with tf.variable_scope(name):
        inputs = tf.layers.conv2d(inputs, 256, [1, 1], [1, 1], 'same')
        shape = inputs.get_shape().as_list()
        for v in levels:
            ksize = [1, np.ceil(shape[1] / v + 1).astype(np.int32), np.ceil(shape[2] / v + 1).astype(np.int32), 1]
            strides = [1, np.floor(shape[1] / v + 1).astype(np.int32), np.floor(shape[2] / v + 1).astype(np.int32), 1]

            if pool_type == 'max_pool':
                pool = tf.nn.max_pool(inputs, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool, (shape[0], -1), )
            else:
                pool = tf.nn.avg_pool(inputs, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool, (shape[0], -1))
            if v == 1:
                x_flatten = tf.reshape(pool, (shape[0], -1))
            else:
                x_flatten = tf.concat((x_flatten, pool), axis=1)

    return x_flatten


def stochastic_pooling(inputs, window_size, strides, padding, is_train):
    '''
    The implementation of stochastic pooling.
    :param inputs: input tensor with shape [batch size, height, width, channels]
    :param window_size: the size of pooling window, array like. samples:[2,2]
    :param strides: tuple with integers. sample:(2,2)
    :param padding: padding mode,valid of same
    :param is_train: is train or not
    :return: pooled tensor
    '''
    b, r, c, channel = int(inputs.shape[0]), int(inputs.shape[1]), int(inputs.shape[2]), int(inputs.shape[3])
    pr, pc = window_size
    sr, sc = strides
    # compute number of windows
    num_r = math.ceil(r / sr) if padding == 'SAME' else r // sr
    num_c = math.ceil(c / sc) if padding == 'SAME' else c // sc

    # ensure inputs is positive
    condition = tf.less(inputs, 0)
    inputs = tf.where(condition, inputs * 0, inputs)

    if tf.reduce_sum(inputs) == 0:
        inputs = np.random.rand(inputs.shape)

    input_shape = inputs.shape
    batch = input_shape[0]

    # reshape
    w = tf.transpose(inputs, (0, 3, 1, 2))
    w = tf.reshape(w, (batch * channel, r, c))

    def pool(x):
        if tf.reduce_sum(x) == 0:
            cache = tf.zeros((pr, pc))
            cache = tf.assign(cache[0, 0], 1.0)
            return cache

        x_prob = x / tf.reduce_sum(x)

        if is_train:
            # sort from large to small
            size = pr * pc
            x_prob = tf.reshape(x_prob, [size])
            x_sorted = tf.argsort(-x_prob)  # get indices

            h = random.randint(0, size - 1)
            p = x_sorted[h]
            row = tf.to_int32(p // pr)
            column = p % pc

            one_hot = tf.one_hot(row * pr + column, size)
            pool_matrix = tf.reshape(one_hot, (pr, pc))

            return pool_matrix
        else:
            return x_prob

    # re = tf.zeros((batch * channel, num_r, num_c),dtype=tf.float32)
    # extract with pool_size
    initial = tf.zeros((batch * channel, 1, 1))
    for i in tqdm(range(num_r)):
        for j in range(num_c):
            # crop = tf.zeros((batch * channel, pr, pc))
            crop = w[:, i * sr:i * sr + pr, j * sc:j * sc + pc]

            # pool
            outs = tf.map_fn(pool, crop)

            if is_train:
                temp = tf.reduce_max((crop * outs), axis=(1, 2))
                temp = tf.expand_dims(temp, axis=1)
                temp = tf.expand_dims(temp, axis=2)
                initial = tf.concat([initial, temp], axis=-1)
                # re[:, i, j]=0

            else:
                temp = tf.reduce_sum((crop * outs), axis=(1, 2))
                temp = tf.expand_dims(temp, axis=1)
                temp = tf.expand_dims(temp, axis=2)
                initial = tf.concat([initial, temp], axis=-1)

    # reshape
    re = tf.reshape(initial[:, :, 1:], (batch, channel, num_r, num_c))
    re = tf.transpose(re, (0, 2, 3, 1))

    return re


if __name__ == "__main__":
    vector = tf.get_variable('input', shape=[1, 224, 224, 3])
    vector = test_net(vector)

    a = 1
