% Copyright 2016 Abhinav Agarwalla. All Rights Reserved.
% Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
% ==============================================================================

function [levels, width, height] = get_info(image_path)
    a = imfinfo(image_path, 'JP2');
    levels = a.WaveletDecompositionLevels;
    width = a.Width;
    height = a.Height;
end