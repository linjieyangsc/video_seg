#Embedded file name: /raid/ljyang/work/OSVOS-TensorFlow/osmn_sim.py
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
slim = tf.contrib.slim

def osmn_arg_scope(weight_decay = 0.0002):
    """Defines the OSMN arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose], activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.001), weights_regularizer=slim.l2_regularizer(weight_decay), biases_initializer=tf.zeros_initializer(), biases_regularizer=None, padding='SAME') as arg_sc:
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
    slice_input = tf.slice(feature, (0,
     ini_w,
     ini_h,
     0), (-1,
     out_size[1],
     out_size[2],
     -1))
    return tf.reshape(slice_input, [int(feature.get_shape()[0]),
     out_size[1],
     out_size[2],
     int(feature.get_shape()[3])])


def osmn(inputs, n_modulator_param = 64, scope = 'osmn'):
    """Defines the OSMN
    Args:
    inputs: Tensorflow placeholder that contains the input image and the first frame masked forground
    scope: Scope name for the network
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """
    guide_im_size = tf.shape(inputs[0])
    im_size = tf.shape(inputs[1])
    with tf.variable_scope(scope, [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        with tf.variable_scope('modulator'):
            with slim.arg_scope([slim.conv2d], padding='SAME', outputs_collections=end_points_collection):
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
                    net_p = tf.reduce_mean(net_5, [1, 2])
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                fc = slim.fully_connected(net_p, 512, scope='fc')
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.sigmoid, biases_initializer=tf.constant_initializer(5)):
                modulator_params = slim.fully_connected(fc, n_modulator_param, scope='fc2')
        with tf.variable_scope('seg'):
            with slim.arg_scope([slim.conv2d], trainable=False, padding='SAME', outputs_collections=end_points_collection):
                with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                    net = slim.repeat(inputs[1], 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net_2 = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net_2, [2, 2], scope='pool2')
                    net_3 = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net_3, [2, 2], scope='pool3')
                    net_4 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net_4, [2, 2], scope='pool4')
                    net_5 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    with slim.arg_scope([slim.conv2d], activation_fn=None):
                        side_2 = slim.conv2d(net_2, 16, [3, 3], scope='conv2_2_16')
                        side_3 = slim.conv2d(net_3, 16, [3, 3], scope='conv3_3_16')
                        side_4 = slim.conv2d(net_4, 16, [3, 3], scope='conv4_3_16')
                        side_5 = slim.conv2d(net_5, 16, [3, 3], scope='conv5_3_16')
                        with slim.arg_scope([slim.convolution2d_transpose], activation_fn=None, biases_initializer=None, padding='VALID', outputs_collections=end_points_collection, trainable=False):
                            side_2_f = slim.convolution2d_transpose(side_2, 16, 4, 2, scope='score-multi2-up')
                            side_2_f = crop_features(side_2_f, im_size)
                            side_3_f = slim.convolution2d_transpose(side_3, 16, 8, 4, scope='score-multi3-up')
                            side_3_f = crop_features(side_3_f, im_size)
                            side_4_f = slim.convolution2d_transpose(side_4, 16, 16, 8, scope='score-multi4-up')
                            side_4_f = crop_features(side_4_f, im_size)
                            side_5_f = slim.convolution2d_transpose(side_5, 16, 32, 16, scope='score-multi5-up')
                            side_5_f = crop_features(side_5_f, im_size)
                        concat_side = tf.concat([side_2_f,
                         side_3_f,
                         side_4_f,
                         side_5_f], axis=3)
                        concat_side = conditional_normalization(concat_side, modulator_params, scope='concat')
                        with slim.arg_scope([slim.conv2d], trainable=False, normalizer_fn=None):
                            net = slim.conv2d(concat_side, 1, [1, 1], scope='upscore-fuse')
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return (net, end_points)


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def modulator_surgery(variables):
    mod_tensors = []
    for v in variables:
        if 'modulator' in v.name and 'conv' in v.name:
            print v.name
            mod_tensors.append(tf.assign(v, slim.get_model_variables(v.name.replace('modulator', 'seg'))[0]))

    return mod_tensors


def interp_surgery(variables):
    interp_tensors = []
    for v in variables:
        if '-up' in v.name:
            h, w, k, m = v.get_shape()
            tmp = np.zeros((m,
             k,
             h,
             w))
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


def preprocess_img(image):
    """Preprocess the image to adapt it to network requirements
    Args:
    Image we want to input the network (W,H,3) numpy array
    Returns:
    Image ready to input the network (1,W,H,3)
    """
    if type(image) is not np.ndarray:
        image = np.array(Image.open(image), dtype=np.uint8)
    in_ = image[:, :, ::-1]
    in_ = np.subtract(in_, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
    in_ = np.expand_dims(in_, axis=0)
    return in_


def preprocess_labels(label):
    """Preprocess the labels to adapt them to the loss computation requirements
    Args:
    Label corresponding to the input image (W,H) numpy array
    Returns:
    Label ready to compute the loss (1,W,H,1)
    """
    if type(label) is not np.ndarray:
        label = np.array(Image.open(label).split()[0], dtype=np.uint8)
    max_mask = np.max(label) * 0.5
    label = np.greater(label, max_mask)
    label = np.expand_dims(np.expand_dims(label, axis=0), axis=3)
    return label


def masked_image(image, label):
    masked = np.zeros(image.shape)
    label = label[:, :, :, 0]
    for ch in range(image.shape[3]):
        masked[label > 0, ch] = image[label > 0, ch]

    return masked


def preprocess_guide(image, label, border_pixels = 8):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    h, w = label.shape
    ymin = max(0, ymin - border_pixels)
    ymax = min(h - 1, ymax + border_pixels)
    xmin = max(0, xmin - border_pixels)
    xmax = min(w - 1, xmax + border_pixels)
    image = image[ymin:ymax + 1, xmin:xmax + 1]
    label = label[ymin:ymax + 1, xmin:xmax + 1]
    image = preprocess_img(image)
    label = preprocess_labels(label)
    return masked_image(image, label)


def load_vgg_fg_model(ckpt_path):
    """Initialize the network parameters from a general-purpose foreground segmentation model
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initilaizes the network
    """
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_corresp = dict()
    for v in var_to_shape_map:
        if 'conv' in v or 'upscore' in v:
            vars_corresp[v] = slim.get_model_variables(v.replace('osvos', 'osmn/seg'))[0]

    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, vars_corresp)
    return init_fn


def load_vgg_imagenet(ckpt_path):
    """Initialize the network parameters from the VGG-16 pre-trained model provided by TF-SLIM
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initializes the network
    """
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_corresp = dict()
    for v in var_to_shape_map:
        if 'conv' in v:
            vars_corresp[v] = slim.get_model_variables(v.replace('vgg_16', 'osvos'))[0]

    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, vars_corresp)
    return init_fn


def class_balanced_cross_entropy_loss(output, label):
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
    loss_val = tf.multiply(output, labels - output_gt_zero) - tf.log(1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))
    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))
    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg
    return final_loss


def class_balanced_cross_entropy_loss_theoretical(output, label):
    """Theoretical version of the class balanced cross entropy loss to train the network (Produces unstable results)
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """
    output = tf.nn.sigmoid(output)
    labels_pos = tf.cast(tf.greater(label, 0), tf.float32)
    labels_neg = tf.cast(tf.less(label, 1), tf.float32)
    num_labels_pos = tf.reduce_sum(labels_pos)
    num_labels_neg = tf.reduce_sum(labels_neg)
    num_total = num_labels_pos + num_labels_neg
    loss_pos = tf.reduce_sum(tf.multiply(labels_pos, tf.log(output + 1e-05)))
    loss_neg = tf.reduce_sum(tf.multiply(labels_neg, tf.log(1 - output + 1e-05)))
    final_loss = -num_labels_neg / num_total * loss_pos - num_labels_pos / num_total * loss_neg
    return final_loss


def _train(dataset, initial_ckpt, learning_rate, logs_path, max_training_iters, save_step, display_step, global_step, iter_mean_grad = 1, batch_size = 1, momentum = 0.9, resume_training = False, config = None, finetune = 1, test_image_path = None, ckpt_name = 'osmn'):
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
    model_name = os.path.join(logs_path, ckpt_name + '.ckpt')
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)
    guide_image = tf.placeholder(tf.float32, [batch_size,
     None,
     None,
     3])
    input_image = tf.placeholder(tf.float32, [batch_size,
     None,
     None,
     3])
    input_label = tf.placeholder(tf.float32, [batch_size,
     None,
     None,
     1])
    with slim.arg_scope(osmn_arg_scope()):
        net, end_points = osmn([guide_image, input_image])
    if finetune == 0:
        init_weights = load_vgg_imagenet(initial_ckpt)
    else:
        init_weights = load_vgg_fg_model(initial_ckpt)
    with tf.name_scope('losses'):
        main_loss = class_balanced_cross_entropy_loss(net, input_label)
        tf.summary.scalar('main_loss', main_loss)
        output_loss = main_loss
        total_loss = output_loss + tf.add_n(tf.losses.get_regularization_losses())
        tf.summary.scalar('total_loss', total_loss)
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
                grad_accumulator_ops.append(grad_acc.apply_grad(var_grad, local_step=global_step))

        with tf.name_scope('take_gradients'):
            mean_grads_and_vars = []
            for var_ind, grad_acc in grad_accumulator.iteritems():
                mean_grads_and_vars.append((grad_acc.take_grad(iter_mean_grad), grads_and_vars[var_ind][1]))

            apply_gradient_op = optimizer.apply_gradients(mean_grads_and_vars, global_step=global_step)
    merged_summary_op = tf.summary.merge_all()
    if test_image_path is not None:
        probabilities = tf.nn.sigmoid(net)
        img_summary = tf.summary.image('Output probabilities', probabilities, max_outputs=1)
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        print 'Init variable'
        sess.run(init)
        tvars = tf.trainable_variables()
        print 'trainable variables'
        for var in tvars:
            print var.name

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=None)
        last_ckpt_path = tf.train.latest_checkpoint(logs_path)
        if last_ckpt_path is not None and resume_training:
            print 'Initializing from previous checkpoint...'
            saver.restore(sess, last_ckpt_path)
            step = global_step.eval() + 1
        else:
            if finetune == 0:
                print 'Initializing from pre-trained imagenet model...'
                init_weights(sess)
            else:
                print 'Initializing from foreground model...'
                init_weights(sess)
                sess.run(modulator_surgery(slim.get_model_variables()))
            step = 1
        sess.run(interp_surgery(tf.global_variables()))
        print 'Weights initialized'
        print 'Start training'
        while step < max_training_iters + 1:
            for _ in range(0, iter_mean_grad):
                batch_g_image, batch_g_label, batch_image, batch_label = dataset.next_batch(batch_size, 'train')
                image = preprocess_img(batch_image[0])
                label = preprocess_labels(batch_label[0])
                masked_g_image = preprocess_guide(batch_g_image[0], batch_g_label[0])
                run_res = sess.run([total_loss, merged_summary_op] + grad_accumulator_ops, feed_dict={guide_image: masked_g_image,
                 input_image: image,
                 input_label: label})
                batch_loss = run_res[0]
                summary = run_res[1]

            sess.run(apply_gradient_op)
            summary_writer.add_summary(summary, step)
            if step % display_step == 0:
                print >> sys.stderr, '{} Iter {}: Training Loss = {:.4f}'.format(datetime.now(), step, batch_loss)
            if step % save_step == 0:
                if test_image_path is not None:
                    curr_output = sess.run(img_summary, feed_dict={input_image: preprocess_img(test_image_path)})
                    summary_writer.add_summary(curr_output, step)
                save_path = saver.save(sess, model_name, global_step=global_step)
                print 'Model saved in file: %s' % save_path
            step += 1

        if (step - 1) % save_step != 0:
            save_path = saver.save(sess, model_name, global_step=global_step)
            print 'Model saved in file: %s' % save_path
        print 'Finished training.'


def train_finetune(dataset, initial_ckpt, learning_rate, logs_path, max_training_iters, save_step, display_step, global_step, iter_mean_grad = 1, batch_size = 1, momentum = 0.9, resume_training = False, config = None, instance_norm = False, test_image_path = None, ckpt_name = 'osmn'):
    """Finetune OSVOS
    Args:
    See _train()
    Returns:
    """
    finetune = 1
    _train(dataset, initial_ckpt, learning_rate, logs_path, max_training_iters, save_step, display_step, global_step, iter_mean_grad, batch_size, momentum, resume_training, config, finetune, test_image_path, ckpt_name)


def test(dataset, checkpoint_file, result_path, config = None):
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
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)
    batch_size = 1
    guide_image = tf.placeholder(tf.float32, [batch_size,
     None,
     None,
     3])
    input_image = tf.placeholder(tf.float32, [batch_size,
     None,
     None,
     3])
    with slim.arg_scope(osmn_arg_scope()):
        net, end_points = osmn([guide_image, input_image])
    probabilities = tf.nn.sigmoid(net)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver([ v for v in tf.global_variables() if '-up' not in v.name ])
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(interp_surgery(tf.global_variables()))
        saver.restore(sess, checkpoint_file)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        for frame in range(0, dataset.get_test_size()):
            guide_img, guide_label, img, curr_img = dataset.next_batch(batch_size, 'test')
            curr_frame = curr_img[0].split('/')[-1].split('.')[0] + '.png'
            path_fds = curr_img[0].split('/')
            save_path = os.path.join(result_path, path_fds[-3], path_fds[-2])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image = preprocess_img(img[0])
            masked_g_image = preprocess_guide(guide_img[0], guide_label[0])
            res = sess.run(probabilities, feed_dict={guide_image: masked_g_image,
             input_image: image})
            res_np = res.astype(np.float32)[0, :, :, 0] > 0.5
            scipy.misc.imsave(os.path.join(save_path, curr_frame), res_np.astype(np.float32))
            print 'Saving ' + os.path.join(save_path, curr_frame)
            curr_score_name = curr_frame[:-4]
            np.save(os.path.join(save_path, curr_score_name), res.astype(np.float32)[0, :, :, 0])
            print 'Saving ' + os.path.join(save_path, curr_score_name) + '.npy'
