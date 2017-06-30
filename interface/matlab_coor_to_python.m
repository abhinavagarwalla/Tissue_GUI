% Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
% Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
% ==============================================================================
% MATLAB wrapper for converting MATLAB coordinates into python (for nuclei positions)

clear all; clc;

coor = [];
load('1334-12Ii_A1H&E_1.mat')
for i = 1 : size(tile,2)
    if ~isempty(tile(i).detection)
        i
        offsetX = tile(i).rowIdx(1)-1;
        offsetY = tile(i).colIdx(1)-1;
        det = tile(i).detection;
        det = det - [500, 500];
        det = det(:, [2, 1]);
        det(:, 1) = -det(:, 1);
        det = det + [500, 500];
        det(:, 1) = 1000 - det(:, 1);
        det = det + [offsetX, offsetY];
        coor = [coor ;[det tile(i).prediction]];
    end
end