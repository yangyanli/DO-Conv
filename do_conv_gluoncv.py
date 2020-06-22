# coding=utf-8

import mxnet as mx
from mxnet.gluon import nn


class DOConv2D(nn.Conv2D):
    """
           DOConv2D can be used as an alternative for mxnet.gluon.nn.Conv2D.
           The interface is similar to that of Conv2D, with three exceptions:
               1. in_channels: should be set explicitly, otherwise, some tensor
                shapes cannot be correctly inferred by GluonCV.
               2. groups: the parameter to switch between DO-Conv (groups=1),
                DO-DConv (groups=in_channels), DO-GConv (otherwise).
               3. do_conv_dtype: should be set explicitly (with default to float32)
                to match with the network.
            Note that the current DOConv2D implementation in GluonCV only supports D_mul=M*N.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_size,
            strides=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            layout='NCHW',
            activation=None,
            use_bias=True,
            weight_initializer=None,
            bias_initializer='zeros',
            do_conv_dtype='float32',
            **kwargs):
        super(DOConv2D, self).__init__(
            channels=channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            layout=layout,
            activation=activation,
            use_bias=use_bias,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            in_channels=in_channels,
            **kwargs)

        self.in_channels = in_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.do_conv_dtype = do_conv_dtype
        self.groups = groups

        if layout == 'NCHW':
            if isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
                self.kernel_shape = (kernel_size[0], kernel_size[1], in_channels // self.groups, channels)
            else:
                self.kernel_shape = (kernel_size, kernel_size, in_channels // self.groups, channels)
        else:
            raise ValueError('DO-Conv Unsupport layout `%s!`' % layout)

        ###################### Initailization of D ###################################
        self.W_shape = self.kernel_shape
        M = self.W_shape[0]
        N = self.W_shape[1]
        self.D_mul = M * N
        self.in_channel = self.W_shape[2] * self.groups

        if M * N > 1:
            D_name = 'D'
            D_shape = (M * N, self.D_mul, self.in_channel)
            self.D = self.params.get(name=D_name, shape=D_shape, init=mx.init.Zero())
        #####################################################################################

    def hybrid_forward(self, F, x, weight, bias=None, **kwargs):
        M = self.W_shape[0]
        N = self.W_shape[1]

        if 'D' in kwargs.keys():
            D = kwargs['D']

            if isinstance(x, mx.ndarray.ndarray.NDArray):  # apply depthwise_over_parameterization with return of D_X_W_kernel (ndarray)
                # kernel in mxnet is (out_channel, in_channel, M, N), transpose to (M, N, in_channel, out_channel) (tf format)
                W = mx.nd.transpose(weight, axes=(2, 3, 1, 0))

                D_total = D + mx.nd.eye(M * N, dtype=self.do_conv_dtype).reshape((M * N, M * N, 1))  # (M * N, D_mul, in_channel)
                W = W.reshape((self.D_mul, self.in_channel, -1))  # (D_mul, in_channel, out_channel//groups)
                ######################### Compute DoW ############################################
                # D (M * N, D_mul, in_channel)
                # W_for_x_D (D_mul, in_channel, out_channel//groups)
                # einsum('msi,sio->mio', D_total, W_for_x_D, name='DoW')
                D_ex = mx.nd.expand_dims(D_total, axis=-1)  # (M * N, D_mul, in_channel, 1)
                W_ex = mx.nd.tile(mx.nd.expand_dims(W, axis=0), reps=(M * N, 1, 1, 1))  # (M * N, D_mul, in_channel, out_channels//groups)
                DoW = mx.nd.sum(D_ex * W_ex, axis=1)
                ##################################################################################
                DoW = DoW.reshape(self.W_shape)  # (M, N, in_channel//groups, out_channels)
                DoW = mx.nd.transpose(DoW, axes=(3, 2, 0, 1))  # (out_channels, in_channel//groups, M, N)

            else:  # isinstance(x, mx.symbol.symbol.Symbol):  # apply depthwise_over_parameterization with return of D_X_W_kernel (symbol)
                W = mx.sym.transpose(weight, axes=(2, 3, 1, 0))
                out_channel = self.W_shape[3] // self.groups

                W = W.reshape((self.D_mul, self.in_channel, -1))  # (D_mul, in_channel, out_channel)
                D_diag = mx.sym.eye(M * N, dtype=self.do_conv_dtype).reshape((M * N, M * N, 1))  # (M * N, M * N, 1)
                D_diag = mx.sym.tile(D_diag, reps=(1, 1, self.in_channel))  # (M * N, M * N, in_channel)
                D_total = D + D_diag  # (M * N, D_mul, in_channel)
                ############################### Compute DoW ######################################
                # einsum('msi,sio->mio', kernel_D_total, W_for_x_D, name='DoW')
                W_ex = mx.sym.tile(mx.sym.expand_dims(W, axis=0),
                                                  reps=(M * N, 1, 1, 1))  # (M * N, D_mul, in_channel, out_channel)
                D_ex = mx.sym.tile(mx.sym.expand_dims(D_total, axis=-1), reps=(1, 1, 1, out_channel))  # (M * N, D_mul, in_channel, out_channel)
                DoW = mx.sym.sum(D_ex * W_ex, axis=1)
                #################################################################################
                DoW = DoW.reshape(self.W_shape)  # (M, N, in_channel//groups, out_channel*goups)
                DoW = mx.sym.transpose(DoW, axes=(3, 2, 0, 1))  # (out_channels, in_channel, M, N)
        else:
            DoW = weight

        return super(DOConv2D, self).hybrid_forward(F, x, DoW, bias)