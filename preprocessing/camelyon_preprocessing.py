# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images of the general dataset."""

import tensorflow as tf
import random
from tensorflow.python.ops import control_flow_ops


class CamelyonPreprocessing():
    def apply_with_random_selector(self, x, func, num_cases):
        """Computes func(x, sel), with sel sampled from [0...num_cases-1].
      
        Args:
          x: input Tensor.
          func: Python function to apply.
          num_cases: Python int32, number of cases to sample sel from.
      
        Returns:
          The result of func(x, sel), where func receives the value of the
          selector as a python integer, but sel is sampled dynamically.
        """
        sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
        # Pass the real x only to one of the func calls.
        return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]

    def distort_color(self, image, color_ordering=0, fast_mode=True, scope=None):
        """Distort the color of a Tensor image.
      
        Each color distortion is non-commutative and thus ordering of the color ops
        matters. Ideally we would randomly permute the ordering of the color ops.
        Rather then adding that level of complication, we select a distinct ordering
        of color ops for each preprocessing thread.
      
        Args:
          image: 3-D Tensor containing single image in [0, 1].
          color_ordering: Python int, a type of distortion (valid values: 0-3).
          fast_mode: Avoids slower ops (random_hue and random_contrast)
          scope: Optional scope for name_scope.
        Returns:
          3-D Tensor color-distorted image on range [0, 1]
        Raises:
          ValueError: if color_ordering not in [0, 3]
        """
        with tf.name_scope(scope, 'distort_color', [image]):
            if fast_mode:
                if color_ordering == 0:
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                else:
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                if color_ordering == 0:
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                elif color_ordering == 1:
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                elif color_ordering == 2:
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                elif color_ordering == 3:
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                else:
                    raise ValueError('color_ordering must be in [0, 3]')

            # The random_* ops do not necessarily clamp.
            return tf.clip_by_value(image, 0.0, 1.0)

    def preprocess_for_train(self, image, s_height, s_width, t_height, t_width, fast_mode=True, scope=None):
        """Distort one image for training a network.
      
        Distorting images provides a useful technique for augmenting the data
        set during training in order to make the network invariant to aspects
        of the image that do not effect the label.
      
        Additionally it would create image_summaries to display the different
        transformations applied to the image.
      
        Args:
          image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
            [0, 1], otherwise it would converted to tf.float32 assuming that the range
            is [0, MAX], where MAX is largest positive representable number for
            int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
          s_height: integer, image source height.
          s_width: integer, image source width.
          t_height: integer, image target height.
          t_width: integer, image target width.
      
          fast_mode: Optional boolean, if True avoids slower transformations (i.e.
            bi-cubic resizing, random_hue or random_contrast).
          scope: Optional scope for name_scope.
        Returns:
          3-D float Tensor of distorted image used for training with range [-1, 1].
        """
        with tf.name_scope(scope, 'distort_image', [image, s_height, s_width, t_height, t_width]):
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)

            # Randomly crop the image
            if s_height > t_height and s_width > t_width:
                offset_height = random.randint(0, s_height - t_height - 1)
                offset_width = random.randint(0, s_width - t_width - 1)
                image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, t_height, t_width)
                tf.summary.image('cropprd_image', tf.expand_dims(image, 0))
            else:
                # Resize the image to the specified target height and target width.
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image, [t_height, t_width], align_corners=False)
                image = tf.squeeze(image, [0])
            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)

            # Randomly flip the image vertically.
            image = tf.image.random_flip_up_down(image)

            tf.summary.image('flipped_image', tf.expand_dims(image, 0))

            # Randomly distort the colors. There are 4 ways to do it.
            distorted_image = self.apply_with_random_selector(
                image,
                lambda x, ordering: self.distort_color(x, ordering, fast_mode),
                num_cases=4)
            # distorted_image = image

            tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))
            distorted_image = tf.subtract(distorted_image, 0.5)
            distorted_image = tf.multiply(distorted_image, 2.0)
            return distorted_image

    def preprocess_for_eval(self, image, s_height, s_width, t_height, t_width, scope=None):
        """Prepare one image for evaluation.
      
        If height and width are specified it would output an image with that size by
        applying resize_bilinear.
      
        If central_fraction is specified it would cropt the central fraction of the
        input image.
      
        Args:
          image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
            [0, 1], otherwise it would converted to tf.float32 assuming that the range
            is [0, MAX], where MAX is largest positive representable number for
            int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
          s_height: integer, image source height.
          s_width: integer, image source width.
          t_height: integer, image target height.
          t_width: integer, image target width.
          scope: Optional scope for name_scope.
        Returns:
          3-D float Tensor of prepared image.
        """
        with tf.name_scope(scope, 'eval_image', [image, s_height, s_width, t_height, t_width]):
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # Crop the central region of the image with an area containing 87.5% of
            # the original image.

            central_fraction = t_height / s_height
            if central_fraction < 1:
                image = tf.image.central_crop(image, central_fraction=central_fraction)

            if t_height and t_width:
                # Resize the image to the specified height and width.
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image, [t_height, t_width],
                                                 align_corners=False)
                image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            return image

    def preprocess_image(self, image, t_height, t_width, s_height=256, s_width=256, is_training=False, fast_mode=False):
        """Pre-process one image for training or evaluation.
      
        Args:
          image: 3-D Tensor [height, width, channels] with the image.
          t_height: integer, image target height.
          t_width: integer, image target width.
          s_height: integer, image source height.
          s_width: integer, image source width.
          is_training: Boolean. If true it would transform an image for train,
            otherwise it would transform it for evaluation.
          fast_mode: Optional boolean, if True avoids slower transformations.
      
        Returns:
          3-D float Tensor containing an appropriately scaled image
        """
        if is_training:
            return self.preprocess_for_train(image, s_height, s_width, t_height, t_width, fast_mode)
        else:
            return self.preprocess_for_eval(image, s_height, s_width, t_height, t_width)
