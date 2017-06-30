% Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
% Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
% 
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
% ==============================================================================

function res = read_region(image_path, level, region)
    res = imread(image_path, 'ReductionLevel', level, 'PixelRegion', {[region(1),region(2)],[region(3),region(4)]});
end