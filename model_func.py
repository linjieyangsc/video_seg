"""
Model initialization functions
"""
import tensorflow as tf
import os
import numpy as np
from ops import conditional_normalization
import mobilenet_v1
slim = tf.contrib.slim
def load_model_from_numpy(sess, ckpt_path, dst_scope):
    weights = np.load(ckpt_path).tolist()
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=dst_scope)
    matched_variables = []
    variable_assigns = []
    for item in var_list:
        item_name = item.name.split('/')[-2]# "osmn/seg/convN" at beginning, ":0" at last
        item_type = item.name.split('/')[-1].split(':')[0] # weights or bias 
        if not item_name in weights.keys():
            print item.name, 'not in weights'
            continue
        matched_variables.append(item_name)
                                        
        sess.run(tf.assign(item, weights[item_name][item_type]))
    print 'matched variables from ckpt', ckpt_path
    print matched_variables

def load_model(ckpt_path, dst_scope):
    """Initialize the network parameters from an existing model with replaced scope names
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initilaizes the network
    """
    if ckpt_path[-4:] == '.npy':
        init_fn = lambda sess: load_model_from_numpy(sess, ckpt_path, dst_scope)
    else:
        reader = tf.train.NewCheckpointReader(ckpt_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        vars_corresp = dict()
        for v in var_to_shape_map:
            v_name = v.split('/') 
            # deeplab model is a special case
            if len(v_name[-1]) == 1:
                if v_name[-1] == 'b':
                    v_new = v + 'iases'
                elif v_name[-1] == 'w':
                    v_new = v + 'eights'
                if v_name[0].startswith('conv'):
                    v_new = 'deeplab/' + v_name[0][:-2] + '/' + v_new
            else:
                v_new = v
            print v_new
            if '/' not in v_new: continue
            src_scope = v_new.split('/')[0]
            corr_var = slim.get_model_variables(v_new.replace(src_scope, dst_scope))
            if len(corr_var) > 0 and var_to_shape_map[v] == corr_var[0].get_shape().as_list():
                vars_corresp[v] = corr_var[0]
        print 'matched variables from ckpt', ckpt_path
        print vars_corresp.keys()
        init_fn = slim.assign_from_checkpoint_fn(
                ckpt_path,
                vars_corresp)
    return init_fn

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

def padding_image(image, out_size):
    im_size = tf.shape(image)
    pad_h = tf.div(tf.subtract(out_size[1], im_size[1]), 2)
    pad_w = tf.div(tf.subtract(out_size[0], im_size[2]), 2)
    paddings = [[0,0],[pad_h, out_size[1] - im_size[1] - pad_h],[pad_w, out_size[0] - im_size[2] - pad_w], [0,0]]
    padded_image = tf.pad(image, paddings, mode='CONSTANT')
    return tf.reshape(padded_image, [int(image.get_shape()[0]), out_size[1], out_size[0], int(image.get_shape()[3])])

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

def modulated_conv_block(net, repeat, channels, dilation=1, scope_id=0, visual_mod_id = 0,
        visual_modulation_params = None,
        spatial_modulation_params = None,
        visual_modulation = False,
        spatial_modulation = False):
    spatial_mod_id = 0
    for i in range(repeat):
        net = slim.conv2d(net, channels, [3,3], rate=dilation, scope='conv{}/conv{}_{}'.format(scope_id, scope_id, i+1))
        if visual_modulation:
            vis_params = tf.slice(visual_modulation_params, [0,visual_mod_id], [-1,channels], name = 'm_param{}'.format(scope_id))
            net = conditional_normalization(net, vis_params, 
                    scope='conv{}/conv{}_{}'.format(scope_id, scope_id, i+1))
            visual_mod_id += channels
        if spatial_modulation:
            sp_params = tf.slice(spatial_modulation_params, 
                    [0, 0, 0, spatial_mod_id], [-1, -1, -1 , channels], 
                    name = 'm_sp_param{}'.format(scope_id))
            net = tf.add(net, sp_params)
            spatial_mod_id += channels
    return net, visual_mod_id

def visual_modulator(guide_image, model_params, scope='osmn', is_training=False):
    """Defines the visual modulator
    Args:
    gudie_image: visual guide image
    model_params: parameters related to model structure
    scope: scope name for the network
    is_training: training or testing
    Returns:
    Tensor of the visual modulation parameters
    """
    mod_early_conv = model_params.mod_early_conv
    n_modulator_param = 512 * 6 + 256 * 3 + mod_early_conv * 384
    with tf.variable_scope(scope, [guide_image]) as sc, slim.arg_scope(osmn_arg_scope()) as arg_sc:
        end_points_collection = sc.name + '_end_points'
        modulator_params = None

        with tf.variable_scope('modulator'):
            # Collect outputs of all intermediate layers.
            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                outputs_collections=end_points_collection):
                net = slim.repeat(guide_image, 2, slim.conv2d, 64, [3, 3], scope='conv1')
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
                net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout7')
                modulator_params = slim.conv2d(net, n_modulator_param, [1, 1],
                        weights_initializer=tf.zeros_initializer(),  
                        biases_initializer=tf.ones_initializer(),
                        activation_fn=None,normalizer_fn=None,scope='fc8')
                modulator_params = tf.squeeze(modulator_params, [1,2])
    return modulator_params

def visual_modulator_lite(guide_image, model_params, scope='osmn', is_training=False):
    """Defines the visual modulator
    Args:
    gudie_image: visual guide image
    model_params: parameters related to model structure
    scope: scope name for the network
    is_training: training or testing
    Returns:
    Tensor of the visual modulation parameters
    """
    mod_early_conv = model_params.mod_early_conv
    n_modulator_param = 1024 + 512 + 256 + 128
    with tf.variable_scope(scope, [guide_image]) as sc, \
            slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=is_training)) as arg_sc:
        modulator_params = None

        # Collect outputs of all intermediate layers.
        modulator_params, end_points = mobilenet_v1.mobilenet_v1(
                guide_image, scope = 'modulator',
                num_classes = n_modulator_param,
                spatial_squeeze = False,
                is_training = is_training)
    return modulator_params
def osmn_lite(inputs, model_params, visual_modulator_params = None, scope='osmn', is_training=False):
    """Defines the OSMN
    Args:
    inputs: Tensorflow placeholder that contains the input image, visual guide, and spatial guide
    model_params: paramters related to the model structure
    visual_modulator_params: if None it will generate new visual modulation parameters using guide image, otherwise
            it will reuse the current paramters.
    scope: Scope name for the network
    is_training: training or testing 
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """
    guide_im_size = tf.shape(inputs[0])
    im_size = tf.shape(inputs[2])
    batch_size = inputs[1].get_shape().as_list()[0]
    use_visual_modulator = model_params.use_visual_modulator
    use_spatial_modulator = model_params.use_spatial_modulator
    train_seg = model_params.train_seg
    n_modulator_param = 1024 + 512 + 256 + 128
    mod_layer_ids = [3, 5, 11, 13]
    output_stride = 32
    batch_norm_params = {
                    'decay': 0.99,
                    'scale': True,
                    'epsilon': 0.001,
                    'updates_collections': None,
                    'is_training': not model_params.fix_bn and is_training
    }
    if use_visual_modulator and visual_modulator_params==None:
        visual_modulator_params = visual_modulator_lite(inputs[0], model_params, scope=scope, is_training = is_training)
    with tf.variable_scope(scope, [inputs]) as sc, slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=is_training)) as arg_sc:
        end_points_collection = sc.name + '_end_points'
        
        # index to mark the current position of the modulation params
        visual_mod_id = 0
        with tf.variable_scope('modulator_sp'):
            with slim.arg_scope([slim.conv2d],
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params=batch_norm_params,
                                  padding='SAME',
                                  outputs_collections=end_points_collection) as bn_arg_sc:
                if not use_spatial_modulator:
                    sp_mod_params = None
                else:
                    ds_mask = slim.avg_pool2d(inputs[1], [4, 4], stride=4, scope='pool4')
                    conv3_att = slim.conv2d(ds_mask, 128, [1,1], scope='conv3')
                    ds_mask = slim.avg_pool2d(ds_mask, [2, 2], scope='pool8')
                    conv5_att = slim.conv2d(ds_mask, 256, [1,1], scope='conv5')
                    ds_mask = slim.avg_pool2d(ds_mask, [2, 2], scope='pool16')
                    conv11_att = slim.conv2d(ds_mask, 512, [1,1], scope='conv11')
                    ds_mask = slim.avg_pool2d(ds_mask, [2, 2], scope='pool32')
                    conv13_att = slim.conv2d(ds_mask, 1024, [1,1], scope='conv13')
                    sp_mod_params = [conv3_att, conv5_att, conv11_att, conv13_att]
        
        # Collect outputs of all intermediate layers.
        net, end_points = mobilenet_v1.mobilenet_v1_base(inputs[2],
                output_stride = output_stride, 
                vis_mod_params = visual_modulator_params,
                sp_mod_params = sp_mod_params,
                mod_layer_ids = mod_layer_ids,
                scope='seg')

        with slim.arg_scope([slim.conv2d],
                            activation_fn=None, normalizer_fn=None):
            net_2 = end_points['Conv2d_3_pointwise']
            net_3 = end_points['Conv2d_5_pointwise']
            net_4 = end_points['Conv2d_11_pointwise']
            net_5 = end_points['Conv2d_13_pointwise']
            side_2 = slim.conv2d(net_2, 16, [3, 3], scope='conv3_16')
            side_3 = slim.conv2d(net_3, 16, [3, 3], scope='conv5_16')
            side_4 = slim.conv2d(net_4, 16, [3, 3], scope='conv11_16')
            side_5 = slim.conv2d(net_5, 16, [3, 3], scope='conv13_16')
            up_size = [im_size[1]/2, im_size[2]/2]
            side_2_f = tf.image.resize_bilinear(side_2, up_size)
            side_3_f = tf.image.resize_bilinear(side_3, up_size)
            side_4_f = tf.image.resize_bilinear(side_4, up_size)
            side_5_f = tf.image.resize_bilinear(side_5, up_size)
            
            net = tf.concat([side_2_f, side_3_f, side_4_f, side_5_f], axis=3)
            net = slim.conv2d(net, 1, [1,1], scope='score')
            net = tf.image.resize_bilinear(net, [im_size[1], im_size[2]])
        #net = slim.conv2d_transpose(net, 1, output_stride * 2, output_stride, normalizer_fn=None, padding="SAME", scope='score-up')
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points
def osmn(inputs, model_params, visual_modulator_params = None, scope='osmn', is_training=False):
    """Defines the OSMN
    Args:
    inputs: Tensorflow placeholder that contains the input image, visual guide, and spatial guide
    model_params: paramters related to the model structure
    visual_modulator_params: if None it will generate new visual modulation parameters using guide image, otherwise
            it will reuse the current paramters.
    scope: Scope name for the network
    is_training: training or testing 
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """
    guide_im_size = tf.shape(inputs[0])
    im_size = tf.shape(inputs[2])
    batch_size = inputs[1].get_shape().as_list()[0]
    mod_early_conv = model_params.mod_early_conv
    use_visual_modulator = model_params.use_visual_modulator
    use_spatial_modulator = model_params.use_spatial_modulator
    train_seg = model_params.train_seg
    n_modulator_param = 512 * 6 + 256 * 3 + mod_early_conv * 384
    num_mod_layers = [2,2,3,3,3]
    batch_norm_params = {
                    'decay': 0.99,
                    'scale': True,
                    'epsilon': 0.001,
                    'updates_collections': None,
                    'is_training': not model_params.fix_bn and is_training
    }
    if use_visual_modulator and visual_modulator_params==None:
        visual_modulator_params = visual_modulator(inputs[0], model_params, scope=scope, is_training = is_training)
    with tf.variable_scope(scope, [inputs]) as sc, slim.arg_scope(osmn_arg_scope()) as arg_sc:
        end_points_collection = sc.name + '_end_points'
        
        # index to mark the current position of the modulation params
        visual_mod_id = 0
        with tf.variable_scope('modulator_sp'):
            with slim.arg_scope([slim.conv2d],
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params=batch_norm_params,
                                  padding='SAME',
                                  outputs_collections=end_points_collection) as bn_arg_sc:
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
                net_1, visual_mod_id = modulated_conv_block(inputs[2], 2, 64,
                        scope_id = 1, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        spatial_modulation_params = conv1_att,
                        visual_modulation = use_visual_modulator and mod_early_conv,
                        spatial_modulation = use_spatial_modulator and mod_early_conv)

                net_2 = slim.max_pool2d(net_1, [2, 2], scope='pool1')
                net_2, visual_mod_id = modulated_conv_block(net_2, 2, 128,
                        scope_id = 2, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        spatial_modulation_params = conv2_att,
                        visual_modulation = use_visual_modulator and mod_early_conv,
                        spatial_modulation = use_spatial_modulator and mod_early_conv)

                net_3 = slim.max_pool2d(net_2, [2, 2], scope='pool2')
                net_3, visual_mod_id = modulated_conv_block(net_3, 3, 256,
                        scope_id = 3, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        spatial_modulation_params = conv3_att,
                        visual_modulation = use_visual_modulator, 
                        spatial_modulation = use_spatial_modulator)
                net_4 = slim.max_pool2d(net_3, [2, 2], scope='pool3')
                net_4, visual_mod_id = modulated_conv_block(net_4, 3, 512,
                        scope_id = 4, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        spatial_modulation_params = conv4_att,
                        visual_modulation = use_visual_modulator, 
                        spatial_modulation = use_spatial_modulator)
                net_5 = slim.max_pool2d(net_4, [2, 2], scope='pool4')
                net_5, visual_mod_id = modulated_conv_block(net_5, 3, 512,
                        scope_id = 5, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
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
                    with slim.arg_scope([slim.conv2d],
                                        trainable=True, normalizer_fn=None):
                        net = slim.conv2d(concat_side, 1, [1, 1], scope='upscore-fuse')

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points

def osmn_deeplab(inputs, model_params, visual_modulator_params=None, scope='osmn', is_training=False):
    """Defines the OSMN with deeplab backbone
    Args:
    inputs: Tensorflow placeholder that contains the input image, visual guide, and spatial guide
    model_params: paramters related to the model structure
    visual_modulator_params: if None it will generate new visual modulation parameters using guide image, otherwise
            it will reuse the current paramters.
    scope: Scope name for the network
    is_training: training or testing 
    Returns:
    net: output tensor of the network
    end_points: dictionary with all tensors of the network
    """
    guide_im_size = tf.shape(inputs[0])
    im_size = tf.shape(inputs[2])
    mod_early_conv = model_params.mod_early_conv
    use_visual_modulator = model_params.use_visual_modulator
    use_spatial_modulator = model_params.use_spatial_modulator
    batch_norm_params = {
                    'decay': 0.99,
                    'scale': True,
                    'epsilon': 0.001,
                    'updates_collections': None,
                    'is_training': not model_params.fix_bn and is_training
                    }
    n_modulator_param = (512 * 6 + 256 * 3) + mod_early_conv * 384
    num_mod_layers = [2,2,3,3,3]
    aligned_size = model_params.aligned_size
    train_seg = model_params.train_seg
    if use_visual_modulator and visual_modulator_params==None:
        visual_modulator_params = visual_modulator(inputs[0], model_params, scope=scope, is_training = is_training)

    with tf.variable_scope(scope, [inputs]) as sc, slim.arg_scope(osmn_arg_scope()) as arg_sc:
        end_points_collection = sc.name + '_end_points'
        # index to mark the current position of the modulation params
        visual_mod_id = 0
        with tf.variable_scope('modulator_sp'):
            with slim.arg_scope([slim.conv2d],
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params=batch_norm_params,
                                  padding='SAME',
                                  outputs_collections=end_points_collection) as bn_arg_sc:
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
                    conv5_att = slim.conv2d(ds_mask, 512 * num_mod_layers[4], [1,1], scope='conv5')
        with tf.variable_scope('seg'):
            if aligned_size:
                image = padding_image(inputs[1], aligned_size)
            else:
                image = inputs[1]
            # Collect outputs of all intermediate layers.
            with slim.arg_scope([slim.conv2d],
                                padding='SAME', trainable=train_seg,
                                outputs_collections=end_points_collection):
              with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                net_1, visual_mod_id = modulated_conv_block(inputs[2], 2, 64,
                        scope_id = 1, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        spatial_modulation_params = conv1_att,
                        visual_modulation = use_visual_modulator and mod_early_conv,
                        spatial_modulation = use_spatial_modulator and mod_early_conv)

                net_2 = slim.max_pool2d(net_1, [2, 2], scope='pool1')
                net_2, visual_mod_id = modulated_conv_block(net_2, 2, 128,
                        scope_id = 2, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        spatial_modulation_params = conv2_att,
                        visual_modulation = use_visual_modulator and mod_early_conv,
                        spatial_modulation = use_spatial_modulator and mod_early_conv)

                net_3 = slim.max_pool2d(net_2, [2, 2], scope='pool2')
                net_3, visual_mod_id = modulated_conv_block(net_3, 3, 256,
                        scope_id = 3, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        spatial_modulation_params = conv3_att,
                        visual_modulation = use_visual_modulator, 
                        spatial_modulation = use_spatial_modulator)
                net_4 = slim.max_pool2d(net_3, [2, 2], scope='pool3')
                net_4, visual_mod_id = modulated_conv_block(net_4, 3, 512,
                        scope_id = 4, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        spatial_modulation_params = conv4_att,
                        visual_modulation = use_visual_modulator, 
                        spatial_modulation = use_spatial_modulator)
                net_5 = slim.max_pool2d(net_4, [2, 2], stride=1, scope='pool4')
                net_5, visual_mod_id = modulated_conv_block(net_5, 3, 512,
                        dilation = 2, scope_id = 5, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        spatial_modulation_params = conv5_att,
                        visual_modulation = use_visual_modulator,
                        spatial_modulation = use_spatial_modulator)
                pool5 = slim.max_pool2d(net_5, [3, 3], stride=1, scope='pool5')
                ## hole = 6
                fc6_1 = slim.conv2d(pool5, 1024, [3, 3], rate=6, scope='fc6_1')
                fc6_1 = slim.dropout(fc6_1, 0.5, is_training=is_training, scope='drop6_1')
                fc7_1 = slim.conv2d(fc6_1, 1024, [1,1], scope='fc7_1')
                fc7_1 = slim.dropout(fc7_1, 0.5, is_training=is_training, scope='drop7_1')
                fc8_voc12_1 = slim.conv2d(fc7_1, 1, [1,1], activation_fn=None, scope='fc8_voc12_1')
                ## hole = 12
                fc6_2 = slim.conv2d(pool5, 1024, [3,3], rate=12, scope='fc6_2')
                fc6_2 = slim.dropout(fc6_2, 0.5, is_training=is_training, scope='drop6_2')
                fc7_2 = slim.conv2d(fc6_2, 1024, [1,1], scope='fc7_2')
                fc7_2 = slim.dropout(fc7_2, 0.5, is_training=is_training, scope='drop7_2')
                fc8_voc12_2 = slim.conv2d(fc7_2, 1, [1,1], activation_fn=None, scope='fc8_voc12_2')
                ## hole = 18
                fc6_3 = slim.conv2d(pool5, 1024, [3,3], rate=18, scope='fc6_3')
                fc6_3 = slim.dropout(fc6_3, 0.5, is_training=is_training, scope='drop6_3')
                fc7_3 = slim.conv2d(fc6_3, 1024, [1,1], scope='fc7_3')
                fc7_3 = slim.dropout(fc7_3, 0.5, is_training=is_training, scope='drop7_3')
                fc8_voc12_3 = slim.conv2d(fc7_3, 1, [1,1], activation_fn=None, scope='fc8_voc12_3')
                ## hole = 24
                fc6_4 = slim.conv2d(pool5, 1024, [3,3], rate=24, scope='fc6_4')
                fc6_4 = slim.dropout(fc6_4, 0.5, is_training=is_training, scope='drop6_4')
                fc7_4 = slim.conv2d(fc6_4, 1024, [1,1], scope='fc7_4')
                fc7_4 = slim.dropout(fc7_4, 0.5, is_training=is_training, scope='drop7_4')
                fc8_voc12_4 = slim.conv2d(fc7_4, 1, [1,1], activation_fn=None, scope='fc8_voc12_4')
                fc8_voc12 = fc8_voc12_1 + fc8_voc12_2 + fc8_voc12_3 + fc8_voc12_4
                with slim.arg_scope([slim.conv2d_transpose],
                        activation_fn=None, biases_initializer=None, padding='VALID',
                        trainable=False):
                    score_full = slim.conv2d_transpose(fc8_voc12, 1, 16, 8, scope='score-up')
                net = crop_features(score_full, im_size)
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points

def osmn_masktrack(inputs, model_params, visual_modulator_params=None, scope='osmn', is_training=False):
    """Defines the OSMN with masktrack backbone
    Args:
    inputs: Tensorflow placeholder that contains the input image, visual guide, and spatial guide
    model_params: paramters related to the model structure
    visual_modulator_params: if None it will generate new visual modulation parameters using guide image, otherwise
            it will reuse the current paramters.
    scope: Scope name for the network
    is_training: training or testing 
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """
    guide_im_size = tf.shape(inputs[0])
    im_size = tf.shape(inputs[1])
    mod_early_conv = model_params.mod_early_conv
    use_visual_modulator = model_params.use_visual_modulator
    n_modulator_param = (512 * 6 + 256 * 3) + mod_early_conv * 384
    aligned_size = model_params.aligned_size
    train_seg = model_params.train_seg
    if use_visual_modulator and visual_modulator_params==None:
        visual_modulator_params = visual_modulator(inputs[0], model_params, scope=scope, is_training = is_training)
    with tf.variable_scope(scope, [inputs]) as sc, slim.arg_scope(osmn_arg_scope()) as arg_sc:
        end_points_collection = sc.name + '_end_points'
        # index to mark the current position of the modulation params
        visual_mod_id = 0
        with tf.variable_scope('seg'):
    	    combined_image = tf.concat([inputs[1], inputs[2]], axis=3)
            if aligned_size:
                image = padding_image(combined_image, aligned_size)
            else:
                image = combined_image
            # Collect outputs of all intermediate layers.
            with slim.arg_scope([slim.conv2d],
                                padding='SAME', trainable=train_seg,
                                outputs_collections=end_points_collection):
              with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                net_1, visual_mod_id = modulated_conv_block(image, 2, 64,
                        scope_id = 1, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        visual_modulation = use_visual_modulator and mod_early_conv)

                net_2 = slim.max_pool2d(net_1, [3, 3], scope='pool1')
                net_2, visual_mod_id = modulated_conv_block(net_2, 2, 128,
                        scope_id = 2, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        visual_modulation = use_visual_modulator and mod_early_conv)

                net_3 = slim.max_pool2d(net_2, [3, 3], scope='pool2')
                net_3, visual_mod_id = modulated_conv_block(net_3, 3, 256,
                        scope_id = 3, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        visual_modulation = use_visual_modulator)
                net_4 = slim.max_pool2d(net_3, [3, 3], scope='pool3')
                net_4, visual_mod_id = modulated_conv_block(net_4, 3, 512,
                        scope_id = 4, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        visual_modulation = use_visual_modulator)
                net_5 = slim.max_pool2d(net_4, [3, 3], stride=1, scope='pool4')
                net_5, visual_mod_id = modulated_conv_block(net_5, 3, 512,
                        dilation=2, scope_id = 5, visual_mod_id = visual_mod_id,
                        visual_modulation_params = visual_modulator_params,
                        visual_modulation = use_visual_modulator)
                pool5 = slim.max_pool2d(net_5, [3, 3], stride=1, scope='pool5')
                ## hole = 6
                fc6_1 = slim.conv2d(pool5, 1024, [3, 3], rate=6, scope='fc6_1')
                fc6_1 = slim.dropout(fc6_1, 0.5, is_training=is_training, scope='drop6_1')
                fc7_1 = slim.conv2d(fc6_1, 1024, [1,1], scope='fc7_1')
                fc7_1 = slim.dropout(fc7_1, 0.5, is_training=is_training, scope='drop7_1')
                fc8_voc12_1 = slim.conv2d(fc7_1, 1, [1,1], activation_fn=None, scope='fc8_1')
                ## hole = 12
                fc6_2 = slim.conv2d(pool5, 1024, [3,3], rate=12, scope='fc6_2')
                fc6_2 = slim.dropout(fc6_2, 0.5, is_training=is_training, scope='drop6_2')
                fc7_2 = slim.conv2d(fc6_2, 1024, [1,1], scope='fc7_2')
                fc7_2 = slim.dropout(fc7_2, 0.5, is_training=is_training, scope='drop7_2')
                fc8_voc12_2 = slim.conv2d(fc7_2, 1, [1,1], activation_fn=None, scope='fc8_2')
                ## hole = 18
                fc6_3 = slim.conv2d(pool5, 1024, [3,3], rate=18, scope='fc6_3')
                fc6_3 = slim.dropout(fc6_3, 0.5, is_training=is_training, scope='drop6_3')
                fc7_3 = slim.conv2d(fc6_3, 1024, [1,1], scope='fc7_3')
                fc7_3 = slim.dropout(fc7_3, 0.5, is_training=is_training, scope='drop7_3')
                fc8_voc12_3 = slim.conv2d(fc7_3, 1, [1,1], activation_fn=None, scope='fc8_3')
                ## hole = 24
                fc6_4 = slim.conv2d(pool5, 1024, [3,3], rate=24, scope='fc6_4')
                fc6_4 = slim.dropout(fc6_4, 0.5, is_training=is_training, scope='drop6_4')
                fc7_4 = slim.conv2d(fc6_4, 1024, [1,1], scope='fc7_4')
                fc7_4 = slim.dropout(fc7_4, 0.5, is_training=is_training, scope='drop7_4')
                fc8_voc12_4 = slim.conv2d(fc7_4, 1, [1,1], activation_fn=None, scope='fc8_4')
                fc8_voc12 = fc8_voc12_1 + fc8_voc12_2 + fc8_voc12_3 + fc8_voc12_4
                with slim.arg_scope([slim.conv2d_transpose],
                        activation_fn=None, biases_initializer=None, padding='VALID',
                        trainable=False):
                    score_full = slim.conv2d_transpose(fc8_voc12, 1, 16, 8, scope='score-up')
                net = crop_features(score_full, im_size)
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
                raise Exception('input + output channels need to be the same')
                
            if h != w:
                raise Exception('filters need to be square')
            up_filter = upsample_filt(int(h))
            tmp[range(m), range(k), :, :] = up_filter
            interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
    return interp_tensors
