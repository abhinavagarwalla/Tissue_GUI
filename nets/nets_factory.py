# Copyright 2016 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from nets import unet_definition, inception_resnet_v2, alexnet

networks_map = {'unet': unet_definition.UNet,
                'inception_resnet_v2': inception_resnet_v2.InceptionResnetV2,
                'alexnet': alexnet.AlexNet,
                }

def get_network_fn(name, images, num_classes=None, is_training=False):
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    cls = networks_map[name]()
    return cls.model(images, num_classes, is_training=is_training)
