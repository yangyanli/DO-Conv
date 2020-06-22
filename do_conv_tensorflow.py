# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import Conv2D
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils


class DOConv2D(Conv2D):
    """
    DOConv2D can be used as an alternative for tf.keras.layers.Conv2D.
    The interface is similar to that of Conv2D, with two exceptions:
        1. D_mul: the depth multiplier for the over-parameterization.
        2. groups: the parameter to switch between DO-Conv (groups=1),
         DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 D_mul=None,
                 groups=1,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DOConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.groups = groups
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        assert (input_channel % self.groups == 0)

        W_shape = (self.D_mul, input_channel // self.groups, self.filters)
        self.W = self.add_weight(
            name='kernel',
            shape=W_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        ####################### Initailization of D ####################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        if M * N > 1:
            D_shape = (M * N, self.D_mul, input_channel)
            self.D = self.add_weight(name='D', shape=D_shape,
                                     initializer='zeros',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     trainable=True,
                                     dtype=self.dtype)
        ################################################################

        self.built = True

    def call(self, inputs, **kwargs):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        # (D_mul, input_channel // groups, filters)
        W_shape = self.W.get_shape().as_list()
        # (M, N, input_channel // groups, filters)
        DoW_shape = (M, N, W_shape[1], W_shape[2])
        if M * N > 1:
            input_channel = W_shape[1] * self.groups

            D_diag = tf.tile(tf.reshape(tf.eye(M * N), (M * N, M * N, 1)),
                             (1, self.D_mul // (M * N), input_channel))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > MxN
                zeros = tf.zeros((M * N, self.D_mul % (M * N), input_channel))
                D_diag = tf.concat([D_diag, zeros], axis=1)

            ######################### Compute DoW #################
            # (M * N, D_mul, input_channel)
            D = self.D + D_diag
            # (D_mul, input_channel, filters // groups)
            W = tf.reshape(self.W, (self.D_mul, input_channel, -1))

            # einsum outputs (M * N, input_channel, filters // groups),
            # which is reshaped to
            # (M, N, input_channel // groups, filters)
            DoW = tf.reshape(tf.einsum('msi,sio->mio', D, W), DoW_shape)
            #######################################################
        else:
            # in this case D_mul == M*N
            # reshape from
            # (D_mul, input_channel // groups, filters)
            # to
            # (M, N, input_channel // groups, filters)
            DoW = tf.reshape(self.W, DoW_shape)

        data_format = conv_utils.convert_data_format(self.data_format, ndim=4)
        outputs = tf.nn.conv2d(inputs, DoW, strides=self.strides,
                               padding=self.padding.upper(),
                               data_format=data_format,
                               dilations=self.dilation_rate)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super(DOConv2D, self).get_config()
        config['groups'] = self.groups
        config['D_mul'] = self.D_mul
        return config
