# The code in this file comes from the following project.
# https://github.com/Jiaxin-Ye/TIM-Net_SER
# Developed by Jiaxin Ye

import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras.layers
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation, Lambda
from tensorflow.keras.layers import LeakyReLU, add, ReLU
from tensorflow.keras.layers import Conv1D, SpatialDropout1D, Dropout
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from typing import List, Tuple
from tensorflow.keras.optimizers import RMSprop

def mul(x):
    return tf.multiply(x[0], x[1])

def makemean(x):
    shape = (K.int_shape(x)[1], K.int_shape(x)[2])
    meantensor = np.ones(shape, dtype=np.float32)
    meantensor = meantensor / 3
    meantensor = K.expand_dims(meantensor, 0)
    meantensor = tf.convert_to_tensor(meantensor)
    return meantensor

# The residual_block code comes from the following project.
# https://github.com/aascode/GM-TCNet

def residual_block(x, s, i, activation, nb_filters, kernel_size, dropout_rate=0, name=''):
    original_x = x
    # 第一级
    # 1.1
    conv_1_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i, padding='causal')(x)
    conv_1_1 = Activation('relu')(conv_1_1)

    conv_s1_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i, padding='causal')(x)
    conv_s1_1 = Activation('relu')(conv_s1_1)
    conv_s1_1 = Lambda(sigmoid)(conv_s1_1)
    output_1_1 = Lambda(lambda x: tf.multiply(x[0], x[1]))([conv_1_1, conv_s1_1])
    # 1.2
    conv_1_2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i, padding='causal')(x)
    conv_1_2 = Activation('relu')(conv_1_2)

    conv_s1_2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i, padding='causal')(x)
    conv_s1_2 = Activation('relu')(conv_s1_2)
    conv_s1_2 = Lambda(sigmoid)(conv_s1_2)
    output_1_2 = Lambda(lambda x: tf.multiply(x[0], x[1]))([conv_1_2, conv_s1_2])
    # 1.3
    conv_1_3 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i, padding='causal')(x)
    conv_1_3 = Activation('relu')(conv_1_3)

    conv_s1_3 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i, padding='causal')(x)
    conv_s1_3 = Activation('relu')(conv_s1_3)
    conv_s1_3 = Lambda(sigmoid)(conv_s1_3)
    output_1_3 = Lambda(lambda x: tf.multiply(x[0], x[1]))([conv_1_3, conv_s1_3])

    output_1 = add([output_1_1, output_1_2, output_1_3])
    templayer = Lambda(makemean)(output_1)
    output_1 = Lambda(lambda x: tf.multiply(x[0], x[1]))([output_1, templayer])

    # 第二级
    # 2.1
    conv_2_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i * 2, padding='causal')(output_1)
    conv_2_1 = Activation('relu')(conv_2_1)

    conv_s2_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i * 2, padding='causal')(output_1)
    conv_s2_1 = Activation('relu')(conv_s2_1)
    conv_s2_1 = Lambda(sigmoid)(conv_s2_1)

    output_2_1 = Lambda(lambda x: tf.multiply(x[0], x[1]))([conv_2_1, conv_s2_1])
    # 2.2
    conv_2_2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i * 2, padding='causal')(output_1)
    conv_2_2 = Activation('relu')(conv_2_2)

    conv_s2_2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i * 2, padding='causal')(output_1)
    conv_s2_2 = Activation('relu')(conv_s2_2)
    conv_s2_2 = Lambda(sigmoid)(conv_s2_2)

    output_2_2 = Lambda(lambda x: tf.multiply(x[0], x[1]))([conv_2_2, conv_s2_2])
    # 2.3
    conv_2_3 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i * 2, padding='causal')(output_1)
    conv_2_3 = Activation('relu')(conv_2_3)

    conv_s2_3 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i * 2, padding='causal')(output_1)
    conv_s2_3 = Activation('relu')(conv_s2_3)
    conv_s2_3 = Lambda(sigmoid)(conv_s2_3)

    output_2_3 = Lambda(lambda x: tf.multiply(x[0], x[1]))([conv_2_3, conv_s2_3])

    output_2 = add([output_2_1, output_2_2, output_2_3])
    templayer1 = Lambda(makemean)(output_2)
    output_2 = Lambda(lambda x: tf.multiply(x[0], x[1]))([output_2, templayer1])
    if original_x.shape[-1] != output_2.shape[-1]:
        original_x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(original_x)
    res_x = tf.keras.layers.add([original_x, output_2])
    return res_x, output_2

class GTCM:
    """双向 Gated Temporal Convolution Module
       Args:
           nb_filters: 使用的卷积层的过滤器数量。
           kernel_size: 卷积层的内核大小。
           nb_stacks: 残差块堆栈数量。
           dilations: 膨胀率列表。
           activation: 激活函数（例如，relu, leaky_relu）。
           dropout_rate: dropout比例。
           use_skip_connections: 是否使用跳过连接。
           return_sequences: 是否返回完整的序列。
    """
    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=None,
                 activation='relu',
                 use_skip_connections=True,
                 dropout_rate=0.1,
                 return_sequences=True,
                 # name='BiDirectionalGTCM'):
                 name='GTCM'):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.activation = activation
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        if not isinstance(nb_filters, int):
            raise Exception()

    def __call__(self, inputs):
        if self.dilations is None:
            self.dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        # 前向输入和反向输入
        forward_input = inputs
        backward_input = K.reverse(inputs, axes=1)

        # 初始卷积
        forward_input = Conv1D(self.nb_filters, 1, padding='causal')(forward_input)
        backward_input = Conv1D(self.nb_filters, 1, padding='causal')(backward_input)

        #forward_skip_connections = []
        #backward_skip_connections = []
        final_skip_connection = []

        for s in range(self.nb_stacks):
            for i in self.dilations:
                forward_output, skip_out_forward = residual_block(forward_input, s, i, self.activation,
                                                              self.nb_filters, self.kernel_size,
                                                              self.dropout_rate, name=self.name + "_forward")
                backward_output, skip_out_backward = residual_block(backward_input, s, i, self.activation,
                                                               self.nb_filters, self.kernel_size,
                                                               self.dropout_rate, name=self.name + "_backward")

                temp_skip = add([skip_out_forward, skip_out_backward], name="biadd_" + str(i))
                temp_skip = GlobalAveragePooling1D()(temp_skip)
                temp_skip = tf.expand_dims(temp_skip, axis=1)
                final_skip_connection.append(temp_skip)

        # 将不同膨胀率的特征组合
        output_2 = final_skip_connection[0]
        for i, item in enumerate(final_skip_connection):
            if i == 0:
                continue
            output_2 = K.concatenate([output_2, item], axis=-2)
        x = output_2

        # 添加激活函数
        #x = LeakyReLU(alpha=0.05)(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1], 1)


        
