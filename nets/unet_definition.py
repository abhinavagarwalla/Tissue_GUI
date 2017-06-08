from dl_interface.model_config import *
from nets.network_ops import *

class UNet():
    def __init__(self):
        pass

    def model(self, images, nclasses=None, is_training=False):
        return self.model_test(images)

    def model_test(self, images):
        with tf.name_scope('Down_Sample_1'):
            x_dim = Config.PATCH_SIZE
            y_dim = Config.PATCH_SIZE
            in_ch_dim = 3
            out_ch_dim = 64
            d_sample_1, x_dim_1, y_dim_1, in_ch_dim = down_sample(images, x_dim, y_dim, in_ch_dim, out_ch_dim, 'DS_1',
                                                                  maxPool=0)

        with tf.name_scope('Down_Sample_2'):
            out_ch_dim = in_ch_dim * 2
            d_sample_2, x_dim_2, y_dim_2, in_ch_dim = down_sample(d_sample_1, x_dim_1, y_dim_1, in_ch_dim, out_ch_dim,
                                                                  'DS_2', maxPool=1)

        with tf.name_scope('Down_Sample_3'):
            out_ch_dim = in_ch_dim * 2
            d_sample_3, x_dim_3, y_dim_3, in_ch_dim = down_sample(d_sample_2, x_dim_2, y_dim_2, in_ch_dim, out_ch_dim,
                                                                  'DS_3', maxPool=1)

        with tf.name_scope('Down_Sample_4'):
            out_ch_dim = in_ch_dim * 2
            d_sample_4, x_dim_4, y_dim_4, in_ch_dim = down_sample(d_sample_3, x_dim_3, y_dim_3, in_ch_dim, out_ch_dim,
                                                                  'DS_4', maxPool=1, dropOut=0.5)

        # with tf.name_scope('Down_Sample_5'):
        # disp('Down_Sample_5')
        # out_ch_dim = in_ch_dim * 2;
        # d_sample_5, x_dim_5, y_dim_5, in_ch_dim = down_sample(d_sample_4, x_dim_4, y_dim_4, in_ch_dim, out_ch_dim, 'DS_5', maxPool = 1, dropOut = 0.5)

        # with tf.name_scope('Up_Sample_4'):
        # disp('Up_Sample_4')
        # out_ch_dim = int(in_ch_dim / 2);
        # u_sample_4, ux_dim_4, uy_dim_4, in_ch_dim = up_sample(d_sample_5, d_sample_4, x_dim_5, y_dim_5, x_dim_4, y_dim_4, in_ch_dim, out_ch_dim, 'US_4')

        with tf.name_scope('Up_Sample_3'):
            out_ch_dim = int(in_ch_dim / 2)
            u_sample_3, ux_dim_3, uy_dim_3, in_ch_dim = up_sample(d_sample_4, d_sample_3, x_dim_4, y_dim_4, x_dim_3,
                                                                  y_dim_3, in_ch_dim, out_ch_dim, 'US_3')

        with tf.name_scope('Up_Sample_2'):
            out_ch_dim = int(in_ch_dim / 2)
            u_sample_2, ux_dim_2, uy_dim_2, in_ch_dim = up_sample(u_sample_3, d_sample_2, ux_dim_3, uy_dim_3, x_dim_2,
                                                                  y_dim_2, in_ch_dim, out_ch_dim, 'US_2')

        with tf.name_scope('Up_Sample_1'):
            out_ch_dim = int(in_ch_dim / 2)
            u_sample_1, ux_dim_1, uy_dim_1, in_ch_dim = up_sample(u_sample_2, d_sample_1, ux_dim_2, uy_dim_2, x_dim_1,
                                                                  y_dim_1, in_ch_dim, out_ch_dim, 'US_1')

        with tf.name_scope('Output_Mask'):
            mask = conv2d_full(u_sample_1, in_ch_dim, 2, 'Output_Mask_Conv', kdim=1)

        with tf.name_scope('SoftMax'):
            eMask = tf.exp(mask)
            sMask = tf.reduce_sum(eMask, 3, keep_dims=True)
            softMask = tf.div(eMask, sMask)

        return softMask