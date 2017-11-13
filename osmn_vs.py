"""
One-Shot Modulater Network
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
import sys
from datetime import datetime
import os
import scipy.misc
from PIL import Image
from ops import conditional_normalization
from model_init import load_model
slim = tf.contrib.slim


def osmn_arg_scope(weight_decay=0.0002):
    """Defines the OSMN arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=None,
                        padding='SAME'):
        with slim.arg_scope([slim.avg_pool2d, slim.max_pool2d],
                padding='SAME') as arg_sc:
            return arg_sc

def crop_features(feature, out_size):
    """Crop the center of a feature map
    Args:
    feature: Feature map to crop
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    """
    up_size = tf.shape(feature)
    ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
    ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
    slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
    # slice_input = tf.slice(feature, (0, ini_w, ini_w, 0), (-1, out_size[1], out_size[2], -1))  # Caffe cropping way
    return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])

def modulated_conv_block(net, repeat, channels, scope_id=0, visual_mod_id = 0,
        trimmed_mod = False,
        visual_modulation_params = None,
        spatial_modulation_params = None,
        visual_modulation = False,
        spatial_modulation = False):
    spatial_mod_id = 0
    for i in range(repeat):
        net = slim.conv2d(net, channels, [3,3], scope='conv{}/conv{}_{}'.format(scope_id, scope_id, i+1))
        if visual_modulation and (not trimmed_mod or i == 0):
            vis_params = tf.slice(visual_modulation_params, [0,visual_mod_id], [-1,channels], name = 'm_param{}'.format(scope_id))
            net = conditional_normalization(net, vis_params, 
                    scope='conv{}/conv{}_{}'.format(scope_id, scope_id, i+1))
            visual_mod_id += channels
        if spatial_modulation and (not trimmed_mod or i==0):
            sp_params = tf.slice(spatial_modulation_params, 
                    [0, 0, 0, spatial_mod_id], [-1, -1, -1 , channels], 
                    name = 'm_sp_param{}'.format(scope_id))
            net = tf.add(net, sp_params)
            spatial_mod_id += channels
    return net, visual_mod_id

def osmn(inputs, model_params, scope='osmn', is_training=False):
    """Defines the OSMN
    Args:
    inputs: Tensorflow placeholder that contains the input image and the first frame masked forground
    scope: Scope name for the network
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """
    guide_im_size = tf.shape(inputs[0])
    im_size = tf.shape(inputs[2])
    batch_size = inputs[1].get_shape().as_list()[0]
    mod_last_conv = model_params.mod_last_conv
    mod_early_conv = model_params.mod_early_conv
    spatial_mod_use_bn = model_params.spatial_mod_use_bn
    orig_gb = model_params.orig_gb
    visual_mod_use_dropout = model_params.visual_mod_use_dropout
    use_visual_modulator = model_params.use_visual_modulator
    use_spatial_modulator = model_params.use_spatial_modulator
    trimmed_mod = model_params.trimmed_mod
    train_seg = model_params.train_seg
    if trimmed_mod:
        n_modulator_param = 512 * 2 + 256 + mod_early_conv * 192 + mod_last_conv * 64
        num_mod_layers = [1,1,1,1,1]
    else:
        n_modulator_param = 512 * 6 + 256 * 3 + mod_early_conv * 384 + mod_last_conv * 64
        num_mod_layers = [2,2,3,3,3]
    batch_norm_params = {
                    'decay': 0.99,
                    'scale': True,
                    'epsilon': 0.001,
                    'updates_collections': None,
                    'is_training': not model_params.fix_bn and is_training
    }
    with tf.variable_scope(scope, [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        modulator_params = None
        if use_visual_modulator:

            with tf.variable_scope('modulator'):
                # Collect outputs of all intermediate layers.
                with slim.arg_scope([slim.conv2d],
                                    padding='SAME',
                                    outputs_collections=end_points_collection):
                  with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                    net = slim.repeat(inputs[0], 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net_2 = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net_2, [2, 2], scope='pool2')
                    net_3 = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net_3, [2, 2], scope='pool3')
                    net_4 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net_4, [2, 2], scope='pool4')
                    net_5 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net_5, [2, 2], scope='pool5')
                    net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                    if visual_mod_use_dropout:
                        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout6')
                    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                    if visual_mod_use_dropout:
                        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout7')
                    #with slim.arg_scope([slim.fully_connected],
                    #        activation_fn = tf.nn.sigmoid, biases_initializer=tf.constant_initializer(5)):
                    modulator_params = slim.conv2d(net, n_modulator_param, [1, 1],
                            weights_initializer=tf.zeros_initializer(),  
                            biases_initializer=tf.ones_initializer(),
                            activation_fn=None,normalizer_fn=None,scope='fc8')
                   # modulator_params = slim.fully_connected(net, n_modulator_param, scope='fc_pred')
                    modulator_params = tf.squeeze(modulator_params, [1,2])
        # index to mark the current position of the modulation params
        visual_mod_id = 0
        if orig_gb:
       
            with tf.variable_scope('modulator_sp'):
                with slim.arg_scope([slim.conv2d],
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=slim.batch_norm,
                                      normalizer_params=batch_norm_params,
                                      outputs_collections=end_points_collection):

                      masknet = slim.repeat(inputs[1], 2, slim.conv2d, 16, [3, 3], scope='conv1')
                      pool1 = slim.avg_pool2d(masknet, [8, 8], scope='pool1', padding='SAME')
                      masknet = slim.repeat(pool1, 2, slim.conv2d, 16, [3, 3], scope='conv2')
                      pool2 = slim.avg_pool2d(masknet, [4, 4], scope='pool2', padding='SAME')
                      masknet = slim.repeat(pool2, 2, slim.conv2d, 32, [3, 3], scope='conv3')
                      pool3 = slim.avg_pool2d(masknet, [4, 4], scope='pool3', padding='SAME')
                      masknet = slim.repeat(pool3, 2, slim.conv2d, 32, [3, 3], scope='conv4')
                      pool4 = slim.avg_pool2d(masknet, [2, 2], scope='pool4', padding='SAME')

            with tf.variable_scope('seg'):
                # Collect outputs of all intermediate layers.
                #net = slim.repeat(inputs[2], 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net_1, visual_mod_id = modulated_conv_block(inputs[2], 2, 64, 
                        scope_id = 1, visual_mod_id = visual_mod_id, 
                        visual_modulation_params = modulator_params, 
                        visual_modulation = use_visual_modulator and mod_early_conv) 
                                                
                net_2 = slim.max_pool2d(net_1, [2, 2], scope='pool1')
                #net_2 = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net_2, visual_mod_id = modulated_conv_block(net_2, 2, 128,
                        scope_id = 2, visual_mod_id = visual_mod_id,
                        visual_modulation_params = modulator_params,
                        visual_modulation = use_visual_modulator and mod_early_conv)
                net_3 = slim.max_pool2d(net_2, [2, 2], scope='pool2')
                net_3, visual_mod_id = modulated_conv_block(net_3, 3, 256,
                        scope_id = 3, visual_mod_id = visual_mod_id,
                        visual_modulation_params = modulator_params,
                        visual_modulation = use_visual_modulator)
                net_4 = slim.max_pool2d(net_3, [2, 2], scope='pool3')
                net_4, visual_mod_id = modulated_conv_block(net_4, 3, 512,
                        scope_id = 4, visual_mod_id = visual_mod_id,
                        visual_modulation_params = modulator_params,
                        visual_modulation = use_visual_modulator)
                net_5 = slim.max_pool2d(net_4, [2, 2], scope='pool4')
                net_5, visual_mod_id = modulated_conv_block(net_5, 3, 512,
                        scope_id = 5, visual_mod_id = visual_mod_id,
                        visual_modulation_params = modulator_params,
                        visual_modulation = use_visual_modulator)
                    
                with slim.arg_scope([slim.conv2d],
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=slim.batch_norm,
                                      normalizer_params=batch_norm_params,
                                      outputs_collections=end_points_collection):

                    with slim.arg_scope([slim.conv2d_transpose],
                                            activation_fn=None, biases_initializer=None, padding='VALID',
                                            outputs_collections=end_points_collection, trainable=False):
                      # upsampling branch 1: 2x
                      mask1 = slim.conv2d(pool1, 8, [3, 3], scope='conv1_0')
                      im2 = slim.conv2d(net_2, 16, [3, 3], scope='conv1_1')
                      branch1 = tf.concat([im2, mask1], axis=3)
                      branch1 = slim.repeat(branch1, 3, slim.conv2d, 16, [3, 3],
                                            scope='conv1_2')
                      branch1 = slim.conv2d_transpose(branch1, num_outputs=16,
                                                    kernel_size=[4, 4], stride=2,
                                                    scope='score1-up')
                      branch1 = crop_features(branch1, im_size)

                      # upsampling branch 2: 4x
                      mask2 = slim.conv2d(pool2, 8, [3, 3], scope='conv2_0')
                      im3 = slim.conv2d(net_3, 16, [3, 3], scope='conv2_1')
                      branch2 = tf.concat([im3, mask2], axis=3)
                      branch2 = slim.repeat(branch2, 3, slim.conv2d, 16, [3, 3],
                                            scope='conv2_2')
                      branch2 = slim.conv2d_transpose(branch2, num_outputs=16,
                                                    kernel_size=[8, 8], stride=4,
                                                    scope='score2-up')
                      branch2 = crop_features(branch2, im_size)

                      # upsampling branch 3: 8x
                      mask3 = slim.conv2d(pool3, 8, [3, 3], scope='conv3_0')
                      im4 = slim.conv2d(net_4, 16, [3, 3], scope='conv3_1')
                      branch3 = tf.concat([im4, mask3], axis=3)
                      branch3 = slim.repeat(branch3, 3, slim.conv2d, 16, [3, 3],
                                            scope='conv3_2')
                      branch3 = slim.conv2d_transpose(branch3, num_outputs=16,
                                                    kernel_size=[16, 16], stride=8,
                                                    scope='score3-up')
                      branch3 = crop_features(branch3, im_size)

                      # upsampling branch 4: 16x
                      mask4 = slim.conv2d(pool4, 8, [3, 3], scope='conv4_0')
                      im5 = slim.conv2d(net_5, 16, [3, 3], scope='conv4_1')
                      branch4 = tf.concat([im5, mask4], axis=3)
                      branch4 = slim.repeat(branch4, 3, slim.conv2d, 16, [3, 3],
                                            scope='conv4_2')
                      branch4 = slim.conv2d_transpose(branch4, num_outputs=16,
                                                    kernel_size=[32, 32], stride=16,
                                                    scope='score4-up')
                      branch4 = crop_features(branch4, im_size)

                      net = tf.concat([branch1, branch2, branch3, branch4],
                                      axis=3, name='concat')

                      height = tf.shape(net)[1]
                      width = tf.shape(net)[2]
                      net = slim.conv2d(net, 1, [1, 1],
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        scope='upscore-fuse')
        else:
            with tf.variable_scope('modulator_sp'):
                if spatial_mod_use_bn:
                    with slim.arg_scope([slim.conv2d],
                                          activation_fn=tf.nn.relu,
                                          normalizer_fn=slim.batch_norm,
                                          normalizer_params=batch_norm_params,
                                          padding='SAME',
                                          outputs_collections=end_points_collection) as bn_arg_sc:
                        pass
                else:
                    with slim.arg_scope([slim.conv2d],
                                        activation_fn=tf.nn.relu) as bn_arg_sc:
                        pass
                with slim.arg_scope(bn_arg_sc):
                    if not use_spatial_modulator:
                        conv1_att = None
                        conv2_att = None
                        conv3_att = None
                        conv4_att = None
                        conv5_att = None
                    else:
                        ds_mask = slim.avg_pool2d(inputs[1], [2, 2], scope='pool1')
                        if mod_early_conv:
                            conv1_att = slim.conv2d(inputs[1], 64 * num_mod_layers[0], [1,1], scope='conv1')
                            conv2_att = slim.conv2d(ds_mask, 128 * num_mod_layers[1], [1,1], scope='conv2')
                        else:
                            conv1_att = None
                            conv2_att = None
                        ds_mask = slim.avg_pool2d(ds_mask, [2,2], scope='pool2')
                        conv3_att = slim.conv2d(ds_mask, 256 * num_mod_layers[2], [1,1], scope='conv3')
                        ds_mask = slim.avg_pool2d(ds_mask, [2, 2], scope = 'pool3')
                        conv4_att = slim.conv2d(ds_mask, 512 * num_mod_layers[3], [1,1], scope='conv4')
                        ds_mask = slim.avg_pool2d(ds_mask, [2, 2], scope = 'pool4')
                        conv5_att = slim.conv2d(ds_mask, 512 * num_mod_layers[4], [1,1], scope='conv5')
            
            with tf.variable_scope('seg'):
                # Collect outputs of all intermediate layers.
                with slim.arg_scope([slim.conv2d],
                                    padding='SAME', trainable = train_seg,
                                    outputs_collections=end_points_collection):
                  with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                    net_1, visual_mod_id = modulated_conv_block(inputs[2], 2, 64,
                            scope_id = 1, visual_mod_id = visual_mod_id,
                            trimmed_mod = trimmed_mod,
                            visual_modulation_params = modulator_params,
                            spatial_modulation_params = conv1_att,
                            visual_modulation = use_visual_modulator and mod_early_conv,
                            spatial_modulation = use_spatial_modulator and mod_early_conv)

                    net_2 = slim.max_pool2d(net_1, [2, 2], scope='pool1')
                    net_2, visual_mod_id = modulated_conv_block(net_2, 2, 128,
                            scope_id = 2, visual_mod_id = visual_mod_id,
                            trimmed_mod = trimmed_mod,
                            visual_modulation_params = modulator_params,
                            spatial_modulation_params = conv2_att,
                            visual_modulation = use_visual_modulator and mod_early_conv,
                            spatial_modulation = use_spatial_modulator and mod_early_conv)

                    net_3 = slim.max_pool2d(net_2, [2, 2], scope='pool2')
                    net_3, visual_mod_id = modulated_conv_block(net_3, 3, 256,
                            scope_id = 3, visual_mod_id = visual_mod_id,
                            trimmed_mod = trimmed_mod,
                            visual_modulation_params = modulator_params,
                            spatial_modulation_params = conv3_att,
                            visual_modulation = use_visual_modulator, 
                            spatial_modulation = use_spatial_modulator)
                    net_4 = slim.max_pool2d(net_3, [2, 2], scope='pool3')
                    net_4, visual_mod_id = modulated_conv_block(net_4, 3, 512,
                            scope_id = 4, visual_mod_id = visual_mod_id,
                            trimmed_mod = trimmed_mod,
                            visual_modulation_params = modulator_params,
                            spatial_modulation_params = conv4_att,
                            visual_modulation = use_visual_modulator, 
                            spatial_modulation = use_spatial_modulator)
                    net_5 = slim.max_pool2d(net_4, [2, 2], scope='pool4')
                    net_5, visual_mod_id = modulated_conv_block(net_5, 3, 512,
                            scope_id = 5, visual_mod_id = visual_mod_id,
                            trimmed_mod = trimmed_mod,
                            visual_modulation_params = modulator_params,
                            spatial_modulation_params = conv5_att,
                            visual_modulation = use_visual_modulator,
                            spatial_modulation = use_spatial_modulator)
                    # Get side outputs of the network
                    with slim.arg_scope([slim.conv2d],
                                        activation_fn=None):
                        side_2 = slim.conv2d(net_2, 16, [3, 3], scope='conv2_2_16')
                        side_3 = slim.conv2d(net_3, 16, [3, 3], scope='conv3_3_16')
                        side_4 = slim.conv2d(net_4, 16, [3, 3], scope='conv4_3_16')
                        side_5 = slim.conv2d(net_5, 16, [3, 3], scope='conv5_3_16')

                        with slim.arg_scope([slim.convolution2d_transpose],
                                            activation_fn=None, biases_initializer=None, padding='VALID',
                                            outputs_collections=end_points_collection, trainable=False):
                            
                            # Main output
                            side_2_f = slim.convolution2d_transpose(side_2, 16, 4, 2, scope='score-multi2-up')
                            side_2_f = crop_features(side_2_f, im_size)
                            side_3_f = slim.convolution2d_transpose(side_3, 16, 8, 4, scope='score-multi3-up')
                            side_3_f = crop_features(side_3_f, im_size)
                            side_4_f = slim.convolution2d_transpose(side_4, 16, 16, 8, scope='score-multi4-up')
                            side_4_f = crop_features(side_4_f, im_size)
                            side_5_f = slim.convolution2d_transpose(side_5, 16, 32, 16, scope='score-multi5-up')
                            side_5_f = crop_features(side_5_f, im_size)
                        concat_side = tf.concat([side_2_f, side_3_f, side_4_f, side_5_f], axis=3)
                        if mod_last_conv:
                            m_params = tf.slice(modulator_params, [0, visual_mod_id], [batch_size, 64], name='m_param_fuse')
                            concat_side = conditional_normalization(concat_side, m_params, scope='conat')

                        with slim.arg_scope([slim.conv2d],
                                            trainable=True, normalizer_fn=None):
                            net = slim.conv2d(concat_side, 1, [1, 1], scope='upscore-fuse')

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# Set deconvolutional layers to compute bilinear interpolation
def interp_surgery(variables):
    interp_tensors = []
    for v in variables:
        if '-up' in v.name:
            h, w, k, m = v.get_shape()
            tmp = np.zeros((m, k, h, w))
            if m != k:
                print 'input + output channels need to be the same'
                raise
            if h != w:
                print 'filters need to be square'
                raise
            up_filter = upsample_filt(int(h))
            tmp[range(m), range(k), :, :] = up_filter
            interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
    return interp_tensors



def class_balanced_cross_entropy_loss(output, label, normalize=False):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = tf.cast(tf.greater(label, 0.5), tf.float32)

    num_labels_pos = tf.reduce_sum(labels)
    num_labels_neg = tf.reduce_sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32)
    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))

    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg
    if normalize:
        final_loss /= num_total

    return final_loss


def binary_seg_summary(images, predictions):
    unnormalized_images = images +  np.array((104, 117, 123)) 
    unnormalized_images = tf.cast(unnormalized_images, tf.float32)
    foreground_predictions = tf.less(0.5, predictions) 
    foreground_labels = tf.cast(foreground_predictions, tf.float32)
    batch_size = images.get_shape().as_list()[0]
    results = tf.add(unnormalized_images * 0.5, foreground_labels * 255 * 0.5)
    results = tf.cast(results, tf.uint8)
    return tf.summary.image('prediction_images', results, batch_size)
    #for idx in range(batch_size):
    #    image = tf.slice(unnormalized_images, [idx,0,0,0],[idx+1,-1,-1,-1])
    #    pred = tf.slice(foregroung_labels, [idx, 0,0,0], [idx+1, -1,-1,1])
    #    results = tf.add(images * 0.6, foreground_labels * 0.4)
    #    tf.summary.image('fgd_prediction_im%d'%idx, results, 3)
def _get_variables_to_train(trainable_scopes):
    """Returns a list of variables to train.
    Returns:
    A list of variables to train by the optimizer.
    """
    if 'ALL' in trainable_scopes:
        print("Trainable Variables: ALL!")
        return tf.trainable_variables()
    variables_to_train = []
    w
    for scope in trainable_scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def _train(dataset, model_params, initial_ckpt, fg_ckpt, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False, config=None, finetune=1,
           use_image_summary=True, ckpt_name="osmn"):
    """Train OSVOS
    Args:
    dataset: Reference to a Dataset object instance
    initial_ckpt: Path to the checkpoint to initialize the network (May be parent network or pre-trained Imagenet)
    supervison: Level of the side outputs supervision: 1-Strong 2-Weak 3-No supervision
    learning_rate: Value for the learning rate. It can be a number or an instance to a learning rate object.
    logs_path: Path to store the checkpoints
    max_training_iters: Number of training iterations
    save_step: A checkpoint will be created every save_steps
    display_step: Information of the training will be displayed every display_steps
    global_step: Reference to a Variable that keeps track of the training steps
    iter_mean_grad: Number of gradient computations that are average before updating the weights
    batch_size: Size of the training batch
    momentum: Value of the momentum parameter for the Momentum optimizer
    resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False)
    config: Reference to a Configuration object used in the creation of a Session
    finetune: Use to select the type of training, 0 for the parent network and 1 for finetunning
    test_image_path: If image path provided, every save_step the result of the network with this image is stored
    Returns:
    """
    model_name = os.path.join(logs_path, ckpt_name+".ckpt")
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True

    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare the input data
    guide_image = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    gb_image = tf.placeholder(tf.float32, [batch_size, None, None, 1])
    input_label = tf.placeholder(tf.float32, [batch_size, None, None, 1])

    # Create the network
    with slim.arg_scope(osmn_arg_scope()):
        net, end_points = osmn([guide_image, gb_image, input_image], model_params, is_training=True)


    # Define loss
    with tf.name_scope('losses'):

        main_loss = class_balanced_cross_entropy_loss(net, input_label, normalize=model_params.loss_normalize)
        tf.summary.scalar('main_loss', main_loss)

        total_loss = main_loss + tf.add_n(tf.losses.get_regularization_losses())
        tf.summary.scalar('total_loss', total_loss)

    # Define optimization method
    with tf.name_scope('optimization'):
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        with tf.name_scope('grad_accumulator'):
            grad_accumulator = {}
            for ind in range(0, len(grads_and_vars)):
                if grads_and_vars[ind][0] is not None:
                    grad_accumulator[ind] = tf.ConditionalAccumulator(grads_and_vars[ind][0].dtype)
        with tf.name_scope('apply_gradient'):
            grad_accumulator_ops = []
            for var_ind, grad_acc in grad_accumulator.iteritems():
                var_name = str(grads_and_vars[var_ind][1].name).split(':')[0]
                var_grad = grads_and_vars[var_ind][0]
                grad_accumulator_ops.append(grad_acc.apply_grad(var_grad,
                                                                local_step=global_step))
        with tf.name_scope('take_gradients'):
            mean_grads_and_vars = []
            for var_ind, grad_acc in grad_accumulator.iteritems():
                mean_grads_and_vars.append(
                    (grad_acc.take_grad(iter_mean_grad), grads_and_vars[var_ind][1]))
            apply_gradient_op = optimizer.apply_gradients(mean_grads_and_vars, global_step=global_step)
    # Log training info
    merged_summary_op = tf.summary.merge_all()

    # Log evolution of test image
    if use_image_summary:
        probabilities = tf.nn.sigmoid(net)
        img_summary = binary_seg_summary(input_image, probabilities)
        #img_summary = tf.summary.image("Output probabilities", probabilities, max_outputs=1)
    # Initialize variables
    init = tf.global_variables_initializer()

    # Create objects to record timing and memory of the graph execution
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) # Option in the session options=run_options
    # run_metadata = tf.RunMetadata() # Option in the session run_metadata=run_metadata
    # summary_writer.add_run_metadata(run_metadata, 'step%d' % i)
    with tf.Session(config=config) as sess:
        print 'Init variable'
        sess.run(init)
        tvars = tf.trainable_variables()
        print 'trainable variables'
        for var in tvars:
            print var.name
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # Create saver to manage checkpoints
        saver = tf.train.Saver(max_to_keep=10)

        last_ckpt_path = tf.train.latest_checkpoint(logs_path)
        if last_ckpt_path is not None and resume_training:
            # Load last checkpoint
            print('Initializing from previous checkpoint...')
            saver.restore(sess, last_ckpt_path)
            step = global_step.eval() + 1
        elif fg_ckpt is not None and len(fg_ckpt) > 0:
            print('Initializing from pre-trained imagenet model...')
            if model_params.use_visual_modulator:
                load_model(initial_ckpt, 'vgg_16', 'osmn/modulator')(sess)
            if 'vgg_16' in fg_ckpt:
                load_model(fg_ckpt, 'vgg_16', 'osmn/seg')(sess)
            else:
                print('Initializing from foreground model...')
                load_model(fg_ckpt, 'osvos', 'osmn/seg')(sess)
            step = 1
        else:
            print('Initializing from pre-trained coco model...')
            load_model(initial_ckpt, 'osmn', 'osmn')(sess)
            #saver.restore(sess, initial_ckpt)
            #sess.run(tf.assign(global_step, 0))
            step = 1
        sess.run(interp_surgery(tf.global_variables()))
        print('Weights initialized')

        print 'Start training'
        while step < max_training_iters + 1:
            # Average the gradient
            for _ in range(0, iter_mean_grad):
                batch_g_image, batch_gb_image, batch_image, batch_label = dataset.next_batch(batch_size, 'train')
                run_res = sess.run([total_loss, merged_summary_op] + grad_accumulator_ops,
                        feed_dict={guide_image: batch_g_image, gb_image: batch_gb_image,
                        input_image: batch_image, input_label: batch_label})
                batch_loss = run_res[0]
                summary = run_res[1]

            # Apply the gradients
            sess.run(apply_gradient_op)  # Momentum updates here its statistics

            # Save summary reports
            summary_writer.add_summary(summary, step)

            # Display training status
            if step % display_step == 0:
                if use_image_summary:
                    #test_g_image, test_gb_image, test_image, _ = dataset.next_batch(batch_size, 'test')
                    curr_output = sess.run(img_summary, feed_dict={guide_image:batch_g_image, gb_image:batch_gb_image,
                    input_image: batch_image })
                    summary_writer.add_summary(curr_output, step)
                print >> sys.stderr, "{} Iter {}: Training Loss = {:.4f}".format(datetime.now(), step, batch_loss)

            # Save a checkpoint
            if step % save_step == 0:
                save_path = saver.save(sess, model_name, global_step=global_step)
                print "Model saved in file: %s" % save_path

            step += 1

        if (step - 1) % save_step != 0:
            save_path = saver.save(sess, model_name, global_step=global_step)
            print "Model saved in file: %s" % save_path

        print('Finished training.')



def train_finetune(dataset, model_params, initial_ckpt, fg_ckpt, learning_rate, logs_path, max_training_iters, save_step,
                   display_step, global_step, iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False,
                   config=None, use_image_summary=True, ckpt_name="osmn"):
    """Finetune OSVOS
    Args:
    See _train()
    Returns:
    """
    finetune = 1
    _train(dataset, model_params, initial_ckpt, fg_ckpt, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad, batch_size, momentum, resume_training, config, finetune, use_image_summary,
           ckpt_name)

def extract_sp_params(model_params, checkpoint_file, result_path, config=None):
    
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)
    batch_size = 1
    guide_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    gb_image = tf.placeholder(tf.float32, [batch_size, None, None, 1])
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])

    # Create the cnn
    with slim.arg_scope(osmn_arg_scope()):
        net, end_points = osmn([guide_image, gb_image, input_image], model_params, is_training=False)
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name]) #if '-up' not in v.name and '-cr' not in v.name])
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)
        with tf.variable_scope("osmn", reuse=True):
            for v in tf.model_variables():
                print v.name
            sp_variables = []
            for layer_id in range(3, 6):
                layer_name = 'modulator_sp/conv%d/weights' % (layer_id)
                v = tf.get_variable(layer_name)
                sp_variables.append(v)
            res = sess.run(sp_variables)
            for layer_id in range(3):
                np.save(os.path.join(result_path, 'sp_params_%d' % (layer_id+3)), 
                        res[layer_id])
        
def test(dataset, model_params, checkpoint_file, result_path, batch_size=1, config=None):
    """Test one sequence
    Args:
    dataset: Reference to a Dataset object instance
    checkpoint_path: Path of the checkpoint to use for the evaluation
    result_path: Path to save the output images
    config: Reference to a Configuration object used in the creation of a Session
    Returns:
    """
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)

    # Input data

    guide_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    gb_image = tf.placeholder(tf.float32, [batch_size, None, None, 1])
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])

    # Create the cnn
    with slim.arg_scope(osmn_arg_scope()):
        net, end_points = osmn([guide_image, gb_image, input_image], model_params, is_training=False)
    probabilities = tf.nn.sigmoid(net)
    #mod_params = end_points['osmn/modulator/fc8']
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name]) #if '-up' not in v.name and '-cr' not in v.name])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(interp_surgery(tf.global_variables()))
        saver.restore(sess, checkpoint_file)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        dataset.reset_idx()
        for frame in range(0, dataset.get_test_size(), batch_size):
            guide_images, gb_images, images, image_paths = dataset.next_batch(batch_size, 'test')
            save_names = [ name.split('.')[0] + '.png' for name in image_paths]
            res_all = sess.run([probabilities], feed_dict={guide_image: guide_images, gb_image:gb_images, input_image: images})
            res = res_all[0]
            #params = res_all[1]
            if model_params.adaptive_crop_testing:
                res = dataset.restore_crop(res)
            if model_params.crf_postprocessing:
                res_np = np.zeros(res.shape[:-1])
                for i in range(batch_size):
                    res_np[i] = dataset.crf_processing(dataset.images[i], (res[i,:,:,0] > 0.5).astype(np.int32))
            else:
                res_np = res.astype(np.float32)[:, :, :, 0] > 0.5             
            for i in range(min(batch_size, dataset.get_test_size() - frame)):
                print 'Saving ' + os.path.join(result_path, save_names[i])
                if len(save_names[i].split('/')) > 1:
                    save_path = os.path.join(result_path, *(save_names[i].split('/')[:-1]))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                scipy.misc.imsave(os.path.join(result_path, save_names[i]), res_np[i].astype(np.float32))
                curr_score_name = save_names[i][:-4]
                if model_params.save_score:
                    print 'Saving ' + os.path.join(result_path, curr_score_name) + '.npy'
                    np.save(os.path.join(result_path, curr_score_name), res.astype(np.float32)[i,:,:,0])
                   # np.save(os.path.join(result_path, curr_score_name+'_params'), params.astype(np.float32)[i])
                #scipy.misc.imsave(os.path.join(result_path, curr_score_name + '_gb.png'), gb_images[i,:,:,0].astype(np.float32))
                #scipy.misc.imsave(os.path.join(result_path, curr_score_name + '_guide.png'),guide_images[i])
