# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Portions of code borrowed from https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition of 2D-LSTM architecture."""

from tensorflow.python.ops import init_ops

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear
from dl_interface.model_config import LSTMTrainConfig
from nets import nets_factory

slim = tf.contrib.slim

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)

def ln(tensor, scope=None, epsilon=1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert (len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    ln_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return ln_initial * scale + shift

def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
  """convolution:
  Args:
    args: a 4D Tensor or a list of 4D, batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    num_features: int, number of features.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 4D Tensor with shape [batch h w num_features]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  print(shapes)
  for shape in shapes:
    if len(shape) != 4:
      raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
    if not shape[3]:
      raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[3]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or "Conv"):
    matrix = tf.get_variable(
        "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
    if len(args) == 1:
      res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
    else:
      res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix, strides=[1, 1, 1, 1], padding='SAME')
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [num_features],
        dtype=dtype,
        initializer=tf.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term

class ConvLSTMCell(RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, filter_size, forget_bias=0.0, activation=tf.nn.tanh):
        self.filter_size = filter_size
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM).
        @param: inputs (batch,n)
        @param state: the states and hidden unit of the two cells
        """
        with tf.variable_scope(scope or type(self).__name__):
            c1, c2, h1, h2 = state

            # change bias argument to False since LN will add bias via shift
            concat = _conv_linear([inputs, h1, h2], self.filter_size, 5 * self._num_units, True)

            i, j, f1, f2, o = tf.split(value=concat, num_or_size_splits=5, axis=3, name='conv_split')

            # add layer normalization to each gate
            # i = ln(i, scope='i/')
            # j = ln(j, scope='j/')
            # f1 = ln(f1, scope='f1/')
            # f2 = ln(f2, scope='f2/')
            # o = ln(o, scope='o/')

            new_c = (c1 * tf.nn.sigmoid(f1 + self._forget_bias) +
                     c2 * tf.nn.sigmoid(f2 + self._forget_bias) +
                     tf.nn.sigmoid(i) *  self._activation(j))

            # add layer_normalization in calculation of new hidden state
            # new_h = self._activation(ln(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)
            new_state = LSTMStateTuple(new_c, new_h)

            return new_h, new_state

class Conv_LSTM_2D():
    def multi_dimensional_rnn_while_loop(self, rnn_size, input_data, context_depth, filter_size, sh, dims=None, scope_n="layer1"):
        """Implements naive multi dimension recurrent neural networks

        @param rnn_size: the hidden units
        @param input_data: the data to process of shape [batch,h,w,channels]
        @param sh: [height,width] of the windows
        @param dims: dimensions to reverse the input data,eg.
            dims=[False,True,True,False] => true means reverse dimension
        @param scope_n : the scope

        returns [batch,h/sh[0],w/sh[1],channels*sh[0]*sh[1]] the output of the lstm
        """
        with tf.variable_scope("ConvLSTMCell-" + scope_n):
            cell = ConvLSTMCell(rnn_size, filter_size=filter_size)

            shape = input_data.get_shape().as_list()

            if shape[1] % sh[0] != 0:
                offset = tf.zeros([shape[0], sh[0] - (shape[1] % sh[0]), shape[2], shape[3]])
                input_data = tf.concat(1, [input_data, offset])
                shape = input_data.get_shape().as_list()
            if shape[2] % sh[1] != 0:
                offset = tf.zeros([shape[0], shape[1], sh[1] - (shape[2] % sh[1]), shape[3]])
                input_data = tf.concat(2, [input_data, offset])
                shape = input_data.get_shape().as_list()

            h, w = int(shape[1] / sh[0]), int(shape[2] / sh[1])
            features = sh[1] * sh[0] * shape[3]
            batch_size = shape[0]

            x = tf.reshape(input_data, [batch_size, h, w, features])
            if dims is not None:
                assert dims[0] is False and dims[3] is False
                x = tf.reshape(x, [1, context_depth, context_depth, h, w, features])
                for i in range(len(dims)):
                    if dims[i]:
                        x = tf.reverse(x, [i])
                x = tf.reshape(x, [batch_size, h, w, features])
            x = [tf.expand_dims(x[i], axis=0) for i in range(shape[0])]
            # x = tf.transpose(x, [1, 2, 0, 3])
            # x = tf.reshape(x, [-1, features])
            # x = tf.split(axis=0, num_or_size_splits=h * w, value=x)

            sequence_length = tf.ones(shape=(batch_size,), dtype=tf.int32) * shape[0]
            inputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='input_ta')
            inputs_ta = inputs_ta.unstack(x)
            states_ta = tf.TensorArray(dtype=tf.float32, size=h * w + 1, name='state_ta', clear_after_read=False)
            outputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='output_ta')

            # initial cell and hidden states
            states_ta = states_ta.write(h * w, LSTMStateTuple(tf.zeros([1, h, w, rnn_size], tf.float32),
                                                              tf.zeros([1, h, w, rnn_size], tf.float32)))

            def get_up(t_, w_):
                return t_ - tf.constant(w_)

            def get_last(t_, w_):
                return t_ - tf.constant(1)

            # def get_up_last(t_, w_):
            #     return t_ - tf.constant(w_) - tf.constant(1)

            time = tf.constant(0)
            zero = tf.constant(0)

            def body(time_, outputs_ta_, states_ta_):
                state_up = tf.cond(tf.less_equal(tf.constant(w), time_),
                                   lambda: states_ta_.read(get_up(time_, w)),
                                   lambda: states_ta_.read(h * w))
                state_last = tf.cond(tf.less(zero, tf.mod(time_, tf.constant(w))),
                                     lambda: states_ta_.read(get_last(time_, w)),
                                     lambda: states_ta_.read(h * w))

                current_state = state_up[0], state_last[0], state_up[1], state_last[1]
                out, state = cell(inputs_ta.read(time_), current_state)
                outputs_ta_ = outputs_ta_.write(time_, out)
                states_ta_ = states_ta_.write(time_, state)
                return time_ + 1, outputs_ta_, states_ta_

            def condition(time_, outputs_ta_, states_ta_):
                return tf.less(time_, tf.constant(h * w))

            result, outputs_ta, states_ta = tf.while_loop(condition, body, [time, outputs_ta, states_ta],
                                                          parallel_iterations=1)

            outputs = outputs_ta.stack()
            states = states_ta.stack()
            states = tf.reshape(states, [-1, 2, h, w, rnn_size])

            y = tf.reshape(outputs, [h, w, batch_size, rnn_size])
            y = tf.transpose(y, [2, 0, 1, 3])
            if dims is not None:
                for i in range(len(dims)):
                    if dims[i]:
                        y = tf.reverse(y, [i])

            return y, states

    def model(self, images, nclasses=None, is_training=False,
              hidden_size_1=int(LSTMTrainConfig.HIDDEN_SIZE/4),
              hidden_size_2=int(LSTMTrainConfig.HIDDEN_SIZE/16)):
        logits, end_points = nets_factory.get_network_fn('alexnet', images, nclasses, is_training)

        fc6input = end_points['alexnet_v2/fc6']
        poolinput = end_points['alexnet_v2/conv5']
        rnn_out_1, _ = self.multi_dimensional_rnn_while_loop(rnn_size=hidden_size_1, input_data=poolinput,
                                                             context_depth=8,
                                                             filter_size=[3,3], sh=[1, 1], scope_n="conv_lstm_1")
        rnn_out_2, _ = self.multi_dimensional_rnn_while_loop(rnn_size=hidden_size_1, input_data=poolinput,
                                                             context_depth=8, filter_size=[3,3], sh=[1, 1],
                                                             dims=[False, True, False, False],
                                                             scope_n="conv_lstm_2")
        rnn_out_3, _ = self.multi_dimensional_rnn_while_loop(rnn_size=hidden_size_1, input_data=poolinput,
                                                             context_depth=8, filter_size=[3,3], sh=[1, 1],
                                                             dims=[False, True, True, False],
                                                             scope_n="conv_lstm_3")
        rnn_out_4, _ = self.multi_dimensional_rnn_while_loop(rnn_size=hidden_size_1, input_data=poolinput,
                                                             context_depth=8, filter_size=[3,3], sh=[1, 1],
                                                             dims=[False, False, True, False],
                                                             scope_n="conv_lstm_4")

        pool_out_1 = slim.max_pool2d(rnn_out_1, [2, 2], scope='conv_pool_1')
        pool_out_2 = slim.max_pool2d(rnn_out_2, [2, 2], scope='conv_pool_2')
        pool_out_3 = slim.max_pool2d(rnn_out_3, [2, 2], scope='conv_pool_3')
        pool_out_4 = slim.max_pool2d(rnn_out_4, [2, 2], scope='conv_pool_4')

        s2_rnn_out_1, _ = self.multi_dimensional_rnn_while_loop(rnn_size=hidden_size_2, input_data=pool_out_1,
                                                             context_depth=8,
                                                             filter_size=[3, 3], sh=[1, 1], scope_n="s2_conv_lstm_1")
        s2_rnn_out_2, _ = self.multi_dimensional_rnn_while_loop(rnn_size=hidden_size_2, input_data=pool_out_2,
                                                             context_depth=8, filter_size=[3, 3], sh=[1, 1],
                                                             dims=[False, True, False, False],
                                                             scope_n="s2_conv_lstm_2")
        s2_rnn_out_3, _ = self.multi_dimensional_rnn_while_loop(rnn_size=hidden_size_2, input_data=pool_out_3,
                                                             context_depth=8, filter_size=[3, 3], sh=[1, 1],
                                                             dims=[False, True, True, False],
                                                             scope_n="s2_conv_lstm_3")
        s2_rnn_out_4, _ = self.multi_dimensional_rnn_while_loop(rnn_size=hidden_size_2, input_data=pool_out_4,
                                                             context_depth=8, filter_size=[3, 3], sh=[1, 1],
                                                             dims=[False, False, True, False],
                                                             scope_n="s2_conv_lstm_4")
        s2_pool_out_1 = slim.max_pool2d(s2_rnn_out_1, [2, 2], scope='s2_conv_pool_1')
        s2_pool_out_2 = slim.max_pool2d(s2_rnn_out_2, [2, 2], scope='s2_conv_pool_2')
        s2_pool_out_3 = slim.max_pool2d(s2_rnn_out_3, [2, 2], scope='s2_conv_pool_3')
        s2_pool_out_4 = slim.max_pool2d(s2_rnn_out_4, [2, 2], scope='s2_conv_pool_4')

        model_out_1 = slim.conv2d(inputs=s2_pool_out_1, num_outputs=nclasses, kernel_size=[3, 3],
                                  activation_fn=None, scope="conv_lstm_conv_1", padding='VALID')
        model_out_2 = slim.conv2d(inputs=s2_pool_out_2, num_outputs=nclasses, kernel_size=[3, 3],
                                  activation_fn=None, scope="conv_lstm_conv_2", padding='VALID')
        model_out_3 = slim.conv2d(inputs=s2_pool_out_3, num_outputs=nclasses, kernel_size=[3, 3],
                                  activation_fn=None, scope="conv_lstm_conv_3", padding='VALID')
        model_out_4 = slim.conv2d(inputs=s2_pool_out_4, num_outputs=nclasses, kernel_size=[3, 3],
                                  activation_fn=None, scope="conv_lstm_conv_4", padding='VALID')

        model_out = tf.scalar_mul(tf.constant(0.25), tf.add_n([model_out_1, model_out_2, model_out_3, model_out_4]))
        return model_out