# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Portion of code borrowed from: Shan, Shaban Camelyon'16 Attempt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helper ops for quickly defining networks"""

import tensorflow as tf

def weight_variable(shape, stddev, name):
    initial = tf.truncated_normal(shape, stddev=stddev, name='Gaussian_Init')
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name='Constant_Init')
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool(x):
    # 2x2 Max Pool
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def activation(x):
    return tf.nn.tanh(x)

def conv2d_full(inTensor, in_ch_dim, out_ch_dim, name_conv, kdim=3):
    """Convolution op"""
    kshape = [kdim, kdim, in_ch_dim, out_ch_dim]
    stddev = tf.sqrt(2 / (kshape[0] * kshape[1] * kshape[2]))

    conv_filter = weight_variable(kshape, stddev=stddev, name=name_conv + '_Filter')
    _ = tf.summary.histogram(name_conv + '_Filter', conv_filter)

    conv_bias = bias_variable([kshape[3]], name=name_conv + '_Bias')
    _ = tf.summary.histogram(name_conv + '_Bias', conv_bias)

    return tf.nn.bias_add(conv2d(inTensor, conv_filter), conv_bias)

def deconv2d_full(inTensor, in_ch_dim, out_ch_dim, name_de_conv, output_shape):
    """Deconvolution op"""
    kshape = [2, 2, in_ch_dim, out_ch_dim]
    stddev = tf.sqrt(2 / (kshape[0] * kshape[1] * kshape[2]))

    de_conv_filter = weight_variable(kshape, stddev=stddev, name=name_de_conv + '_Filter')
    _ = tf.summary.histogram(name_de_conv + '_Filter', de_conv_filter)

    de_conv_bais = bias_variable([kshape[2]], name=name_de_conv + '_Bias')
    _ = tf.summary.histogram(name_de_conv + '_Bias', de_conv_bais)

    return tf.nn.bias_add(
        tf.nn.conv2d_transpose(inTensor, de_conv_filter, output_shape, strides=[1, 2, 2, 1], padding="SAME"),
        de_conv_bais)

def down_sample(inTensor, x_dim, y_dim, in_ch_dim, out_ch_dim, layerID, maxPool=1, dropOut=1):
    """Downsamples the input tensor"""
    with tf.name_scope('Pooling'):
        if maxPool == 1:
            disp('Pooling')
            pool = max_pool(inTensor)
            x_dim = int(x_dim / 2)
            y_dim = int(y_dim / 2)
        else:
            pool = inTensor
    with tf.name_scope('Convolution_1'):
        disp('Convolution_1')

        conv1 = conv2d_full(pool, in_ch_dim, out_ch_dim, layerID + '_Convolution_1')
        x_dim = x_dim - 2
        y_dim = y_dim - 2

    with tf.name_scope('ReLU_1'):
        disp('ReLU_1')
        r_conv1 = activation(conv1)

    with tf.name_scope('Convolution_2'):
        disp('Convolution_2')
        in_ch_dim = out_ch_dim
        conv2 = conv2d_full(r_conv1, in_ch_dim, out_ch_dim, layerID + '_Convolution_2')
        x_dim = x_dim - 2
        y_dim = y_dim - 2

    with tf.name_scope('ReLU_2'):
        disp('ReLU_2')
        r_conv2 = activation(conv2)

    with tf.name_scope('Dropout'):
        disp('Dropout')
        if dropOut < 1:
            outTensor = tf.nn.dropout(r_conv2, dropOut)
        else:
            outTensor = r_conv2
    _ = tf.summary.histogram(layerID + '_Values', outTensor)
    return outTensor, x_dim, y_dim, out_ch_dim

def up_sample(inTensor, symTensor, x_dim, y_dim, sx_dim, sy_dim, in_ch_dim, out_ch_dim, layerID):
    """Upsamples the input tensor"""
    with tf.name_scope('Deconvolution'):
        disp('Deconvolution')
        batch_size = tf.shape(inTensor)[0]
        out_shape = [batch_size, x_dim * 2, y_dim * 2, out_ch_dim]
        deconv = deconv2d_full(inTensor, out_ch_dim, in_ch_dim, layerID + '_Deconvolution', out_shape);

    with tf.name_scope('Concatination'):
        sx_dim = int(sx_dim / 2)
        sy_dim = int(sy_dim / 2)

        croped = symTensor[:, sx_dim - x_dim:sx_dim + x_dim, sy_dim - y_dim:sy_dim + y_dim, :]
        concatinated = tf.concat(axis=3, values=[croped, deconv])
        x_dim = x_dim * 2
        y_dim = y_dim * 2

    with tf.name_scope('Convolution_1'):
        disp('Convolution_1')
        conv1 = conv2d_full(concatinated, in_ch_dim, out_ch_dim, layerID + '_Convolution_1')
        x_dim = x_dim - 2
        y_dim = y_dim - 2

    with tf.name_scope('ReLu_1'):
        disp('ReLu_1')
        r_conv1 = activation(conv1)

    with tf.name_scope('Convolution_2'):
        disp('Convolution_2')
        in_ch_dim = out_ch_dim
        conv2 = conv2d_full(r_conv1, in_ch_dim, out_ch_dim, layerID + '_Convolution_2')
        x_dim = x_dim - 2
        y_dim = y_dim - 2

    with tf.name_scope('ReLu_2'):
        disp('ReLu_2')
        outTensor = activation(conv2)

    _ = tf.summary.histogram(layerID + '_Values', outTensor)
    return outTensor, x_dim, y_dim, out_ch_dim

def disp(msg):
    a = 1 + 1