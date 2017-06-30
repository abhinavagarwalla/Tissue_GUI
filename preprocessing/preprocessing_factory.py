# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the preprocessing factory that loads the required preprocessing function."""

from preprocessing import stain_normalisation, camelyon_preprocessing

preprocessing_map = {'stain_norm': stain_normalisation.StrainNormalisation,
                     'camelyon': camelyon_preprocessing.CamelyonPreprocessing,
               }

def get_preprocessing_fn(name):
    """Returns the requested network.
  
        Args:
          name: Requested preprocessing, must be in preprocessing_map
        
        Returns:
          Class wrapping the requested preprocessing function
    """
    if name not in preprocessing_map:
      raise ValueError('Name of Preprocessing unknown %s' % name)
    return preprocessing_map[name]()