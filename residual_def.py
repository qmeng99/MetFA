from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from residual_unit import vanilla_residual_unit_3d
from dltk.core.upsample import linear_upsample_3d
from dltk.core.activations import leaky_relu


def residual_encoder(inputs,
                     #num_classes,
                     num_res_units=1,
                     filters=(16, 32, 64, 128),
                     strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                     use_bias=False,
                     activation=leaky_relu,
                     mode=tf.estimator.ModeKeys.EVAL,
                     kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
                     bias_initializer=tf.zeros_initializer(),
                     kernel_regularizer=None,
                     bias_regularizer=None):


    assert len(strides) == len(filters)
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'

    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    x = inputs

    # Initial convolution with filters[0]
    x = tf.layers.conv3d(inputs=x,
                         filters=filters[0],
                         kernel_size=(3, 3, 3),
                         strides=strides[0],
                         **conv_params)

    tf.logging.info('Init conv tensor shape {}'.format(x.get_shape()))

    # Residual feature encoding blocks with num_res_units at different
    # resolution scales res_scales
    res_scales = [x]
    saved_strides = []
    for res_scale in range(1, len(filters)):

        # Features are downsampled via strided convolutions. These are defined
        # in `strides` and subsequently saved
        with tf.variable_scope('enc_unit_{}_0'.format(res_scale)):

            x = vanilla_residual_unit_3d(
                inputs=x,
                out_filters=filters[res_scale],
                strides=strides[res_scale],
                activation=activation,
                mode=mode)
        saved_strides.append(strides[res_scale])

        for i in range(1, num_res_units):

            with tf.variable_scope('enc_unit_{}_{}'.format(res_scale, i)):

                x = vanilla_residual_unit_3d(
                    inputs=x,
                    out_filters=filters[res_scale],
                    strides=(1, 1, 1),
                    activation=activation,
                    mode=mode)
        res_scales.append(x)

        tf.logging.info('Encoder at res_scale {} tensor shape: {}'.format(
            res_scale, x.get_shape()))

    return x, res_scales, saved_strides, filters



def classify_dense_bn_relu(x,
                   units=(256,256),
                   is_train=True,
                   use_bias=False,
                   kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
                   bias_initializer=tf.zeros_initializer(),
                   kernel_regularizer=None,
                   bias_regularizer=None):

    conv_params = {'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    # #flatten the middle axis
    # flat_x = tf.contrib.layers.flatten(x)
    # x_new = flat_x

    x_new = x

    for i in range(len(units)):

        dense = tf.layers.dense(inputs=x_new, units=units[i], trainable=is_train, **conv_params)
        bn = tf.contrib.layers.batch_norm(dense, is_training=is_train)
        x_new = tf.nn.relu(bn)

    return x_new

def prototype(x,
           is_train = True,
           num_class = 1,
           use_bias=False,
           kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
           bias_initializer=tf.zeros_initializer(),
           kernel_regularizer=None,
           bias_regularizer=None):

    conv_params = {'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    # #flatten the middle axis
    # flat_x = tf.contrib.layers.flatten(x)
    # x_new = flat_x

    x_new = x

    x_new = tf.nn.l2_normalize(x_new, dim=1)
    logits = tf.layers.dense(inputs=x_new, units=num_class, trainable=is_train, **conv_params)

    logits = logits/0.05  # 0.05 is the temperature

    return logits


def VAE_layer(x,
               outputdim=128,
               is_train = True,
               use_bias=False,
               kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None):

    conv_params = {'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    logits = tf.layers.dense(inputs=x, units=outputdim, trainable=is_train, **conv_params)
    # logits = tf.contrib.layers.batch_norm(dense, is_training=is_train)


    return logits


