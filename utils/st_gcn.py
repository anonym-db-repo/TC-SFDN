import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
from .norm import InstanceNormalization


class STGCN:
    _normalizer = InstanceNormalization  # layers.BatchNormalization

    def gcn(self, x, adjacency_matrix, out_filters, kernel_size, t_kernel_size=1, t_stride=1, t_dilation=1,
            padding='same', bias=True, normalize=True, name=None):
        assert adjacency_matrix.shape[0] == kernel_size
        x = layers.Conv2D(out_filters * kernel_size, kernel_size=(t_kernel_size, 1), padding=padding,
                             strides=(t_stride, 1), dilation_rate=(t_dilation, 1), use_bias=bias, name=name)(x)

        xshape = K.int_shape(x)
        # n = xshape[0]
        t, v, kc = xshape[1:]

        x = layers.Reshape([t, v, kernel_size, kc // kernel_size])(x)  # ntvkc  A:kvw
        # x: [-1, 300, 25, 3, 64]  A: [3, 25, 25]
        x = tf.einsum('ntvkc,kvw->ntwc', x, adjacency_matrix)  # x: [-1, 300, 25, 64]
        # x = tf.nn.relu(x)
        if normalize:
            x = self._normalizer()(x)
        x = layers.LeakyReLU()(x)
        return x

    def causal_gcn(self, x, adjacency_matrix, out_filters, kernel_size, t_kernel_size=1, t_stride=1, t_dilation=1,
                   padding='same', bias=True, normalize=True, name=None):
        x = layers.Conv2D(out_filters * kernel_size, kernel_size=(t_kernel_size, 1), padding=padding,
                             strides=(t_stride, 1), dilation_rate=(t_dilation, 1), use_bias=bias, name=name)(x)
        xshape = K.int_shape(x)
        # n = xshape[0]
        t, v, kc = xshape[1:]

        x = layers.Reshape([t, v, kernel_size, kc // kernel_size])(x)  # ntvkc  A:kvw
        # x: [-1, 50, 25, 1, 64]  A: [-1, 1, 25, 25]
        x = tf.einsum('ntvkc,nkvw->ntwc', x, adjacency_matrix)  # x: [-1, 300, 25, 64]
        if normalize:
            x = self._normalizer()(x)
        x = layers.LeakyReLU()(x)
        return x

    def tcn(self, x, out_filters, kernel_size, stride, drop_rate=0, padding='same', normalize=True, name=None):
        x = layers.Conv2D(out_filters, (kernel_size, 1), (stride, 1), padding=padding, name=name)(x)
        if normalize:
            x = self._normalizer()(x)
        x = layers.LeakyReLU()(x)
        if drop_rate > 0:
            x = layers.Dropout(drop_rate)(x)
        return x

    def residual(self, x, residual, in_filters, out_filters, kernel_size=1, stride=1, padding='same',
                 normalize=True, name=None):
        if not residual:
            res = 0
        elif (in_filters == out_filters) and (stride == 1):
            res = x
        else:
            res = layers.Conv2D(out_filters, kernel_size=(kernel_size, 1), strides=(stride, 1), padding=padding, name=name)(x)
            if normalize:
                res = self._normalizer()(res)
        return res
