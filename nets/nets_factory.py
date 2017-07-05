# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the network factory that loads the required network."""

from nets import unet_definition
from nets import inception_resnet_v2
from nets import alexnet
from nets import two_dimensional_lstm
from nets import stacked_two_dimensional_lstm
from nets import stacked_two_dimensional_lstm_8_connected

networks_map = {'unet': unet_definition.UNet,
                'inception_resnet_v2': inception_resnet_v2.InceptionResnetV2,
                'alexnet': alexnet.AlexNet,
                '2D-LSTM': two_dimensional_lstm.LSTM_2D,
                'Stacked-2D-LSTM': stacked_two_dimensional_lstm.Stacked_LSTM_2D,
                'Stacked-2D-LSTM-8c': stacked_two_dimensional_lstm_8_connected.Stacked_LSTM_2D_8c,
                }


def get_network_fn(name, images, num_classes=None, is_training=False):
    """Returns the requested network.

    Args:
      name: Name of the network to be returned, must be in networks_map
      images: a 4-D tensor of size [batch_size, height, width, 3]
      num_classes: int
      is_training: bool

    Returns:
      model: Requested network, generally output probabilities, and end_points
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    cls = networks_map[name]()
    return cls.model(images, num_classes, is_training=is_training)
