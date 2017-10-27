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
#from models.masktrack import SalSegm_vgg16
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

def modulated_conv_block(net, repeat, channels, dilation=1,
        scope_id=0, visual_mod_id = 0,
        visual_modulation_params = None,
        visual_modulation = False):
    for i in range(repeat):
        net = slim.conv2d(net, channels, [3,3], rate=dilation, scope='conv{}/conv{}_{}'.format(scope_id, scope_id, i+1))
        if visual_modulation:
            vis_params = tf.slice(visual_modulation_params, [0,visual_mod_id], [-1,channels], name = 'm_param{}'.format(scope_id))
            net = conditional_normalization(net, vis_params, 
                    scope='conv{}/conv{}_{}'.format(scope_id, scope_id, i+1))
            visual_mod_id += channels
    return net, visual_mod_id


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
    im_size = tf.shape(inputs[1])
    mod_early_conv = model_params.mod_early_conv
    use_visual_modulator = model_params.use_visual_modulator
    mod_middle_conv = model_params.mod_middle_conv
    n_modulator_param = (512 * 6 + 256 * 3) * mod_middle_conv + mod_early_conv * 384
    aligned_size = model_params.aligned_size
    train_seg = model_params.train_seg
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
                    net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout6')
                    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
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
                net_1, visual_mod_id = modulated_conv_block(image, 2, 64,
                        scope_id = 1, visual_mod_id = visual_mod_id,
                        visual_modulation_params = modulator_params,
                        visual_modulation = use_visual_modulator and mod_early_conv)

                net_2 = slim.max_pool2d(net_1, [3, 3], scope='pool1')
                net_2, visual_mod_id = modulated_conv_block(net_2, 2, 128,
                        scope_id = 2, visual_mod_id = visual_mod_id,
                        visual_modulation_params = modulator_params,
                        visual_modulation = use_visual_modulator and mod_early_conv)

                net_3 = slim.max_pool2d(net_2, [3, 3], scope='pool2')
                net_3, visual_mod_id = modulated_conv_block(net_3, 3, 256,
                        scope_id = 3, visual_mod_id = visual_mod_id,
                        visual_modulation_params = modulator_params,
                        visual_modulation = use_visual_modulator and mod_middle_conv)
                net_4 = slim.max_pool2d(net_3, [3, 3], scope='pool3')
                net_4, visual_mod_id = modulated_conv_block(net_4, 3, 512,
                        scope_id = 4, visual_mod_id = visual_mod_id,
                        visual_modulation_params = modulator_params,
                        visual_modulation = use_visual_modulator and mod_middle_conv)
                net_5 = slim.max_pool2d(net_4, [3, 3], stride=1, scope='pool4')
                net_5, visual_mod_id = modulated_conv_block(net_5, 3, 512,
                        dilation=2, scope_id = 5, visual_mod_id = visual_mod_id,
                        visual_modulation_params = modulator_params,
                        visual_modulation = use_visual_modulator and mod_middle_conv)
                pool5 = slim.max_pool2d(net_5, [3, 3], stride=1, scope='pool5')
                ## hole = 6
                fc6_1 = slim.conv2d(pool5, 1024, [3, 3], rate=6, scope='fc6_1')
                fc6_1 = slim.dropout(fc6_1, 0.5, is_training=is_training, scope='drop6_1')
                fc7_1 = slim.conv2d(fc6_1, 1024, [1,1], scope='fc7_1')
                fc7_1 = slim.dropout(fc7_1, 0.5, is_training=is_training, scope='drop7_1')
                fc8_voc12_1 = slim.conv2d(fc7_1, 2, [1,1], activation_fn=None, scope='fc8_voc12_1')
                ## hole = 12
                fc6_2 = slim.conv2d(pool5, 1024, [3,3], rate=12, scope='fc6_2')
                fc6_2 = slim.dropout(fc6_2, 0.5, is_training=is_training, scope='drop6_2')
                fc7_2 = slim.conv2d(fc6_2, 1024, [1,1], scope='fc7_2')
                fc7_2 = slim.dropout(fc7_2, 0.5, is_training=is_training, scope='drop7_2')
                fc8_voc12_2 = slim.conv2d(fc7_2, 2, [1,1], activation_fn=None, scope='fc8_voc12_2')
                ## hole = 18
                fc6_3 = slim.conv2d(pool5, 1024, [3,3], rate=18, scope='fc6_3')
                fc6_3 = slim.dropout(fc6_3, 0.5, is_training=is_training, scope='drop6_3')
                fc7_3 = slim.conv2d(fc6_3, 1024, [1,1], scope='fc7_3')
                fc7_3 = slim.dropout(fc7_3, 0.5, is_training=is_training, scope='drop7_3')
                fc8_voc12_3 = slim.conv2d(fc7_3, 2, [1,1], activation_fn=None, scope='fc8_voc12_3')
                ## hole = 24
                fc6_4 = slim.conv2d(pool5, 1024, [3,3], rate=24, scope='fc6_4')
                fc6_4 = slim.dropout(fc6_4, 0.5, is_training=is_training, scope='drop6_4')
                fc7_4 = slim.conv2d(fc6_4, 1024, [1,1], scope='fc7_4')
                fc7_4 = slim.dropout(fc7_4, 0.5, is_training=is_training, scope='drop7_4')
                fc8_voc12_4 = slim.conv2d(fc7_4, 2, [1,1], activation_fn=None, scope='fc8_voc12_4')
                fc8_voc12 = fc8_voc12_1 + fc8_voc12_2 + fc8_voc12_3 + fc8_voc12_4
                with slim.arg_scope([slim.conv2d_transpose],
                        activation_fn=None, biases_initializer=None, padding='VALID',
                        trainable=False):
                    score_full = slim.conv2d_transpose(fc8_voc12, 2, 16, 8, scope='score-up')
                net = crop_features(score_full, im_size)
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points
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
    # get the positive score
    output = tf.slice(output, [0,0,0,1],[-1,-1,-1,1]) 
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
    mask_image = tf.placeholder(tf.float32, [batch_size, None, None, 1])
    input_label = tf.placeholder(tf.int32, [batch_size, None, None])
    combined_image = tf.concat([input_image, mask_image], axis=3)

    # Create the network
    with slim.arg_scope(osmn_arg_scope()):
        net, end_points = osmn([guide_image, combined_image], model_params, is_training=True)


    # Define loss
    with tf.name_scope('losses'):

        main_loss = tf.losses.sparse_softmax_cross_entropy(input_label, net)
        #main_loss = class_balanced_cross_entropy_loss(net, input_label, normalize=model_params.loss_normalize)
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
        probabilities = tf.slice(tf.nn.softmax(net),[0,0,0,1],[-1,-1,-1,1])
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
        saver = tf.train.Saver(max_to_keep=1)

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
                print('Initializing from masktrack model...')
                load_model(fg_ckpt, '', 'osmn/seg')(sess)
            step = 1
        else:
            print('Initializing from pre-trained coco model...')
            saver.restore(sess, initial_ckpt)
            sess.run(tf.assign(global_step, 0))
            step = 1
        sess.run(interp_surgery(tf.global_variables()))
        print('Weights initialized')

        print 'Start training'
        while step < max_training_iters + 1:
            # Average the gradient
            for _ in range(0, iter_mean_grad):
                batch_g_image, batch_mask_image, batch_image, batch_label = dataset.next_batch(batch_size, 'train')
                run_res = sess.run([total_loss, merged_summary_op] + grad_accumulator_ops,
                        feed_dict={guide_image: batch_g_image, mask_image: batch_mask_image,
                            input_image: batch_image, input_label: batch_label[:,:,:,0]})
                batch_loss = run_res[0]
                summary = run_res[1]

            # Apply the gradients
            sess.run(apply_gradient_op)  # Momentum updates here its statistics

            # Save summary reports
            summary_writer.add_summary(summary, step)

            # Display training status
            if step % display_step == 0:
                if use_image_summary:
                    test_g_image, test_mask_image, test_image, _ = dataset.next_batch(batch_size, 'test')
                    curr_output = sess.run(img_summary, feed_dict={guide_image:test_g_image, mask_image:test_mask_image,
                    input_image: test_image })
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
    im_size = dataset.train_img_size()
    guide_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    mask_image = tf.placeholder(tf.float32, [batch_size, im_size[1], im_size[0], 1])
    input_image = tf.placeholder(tf.float32, [batch_size, im_size[1], im_size[0], 3])
    combined_image = tf.concat([input_image, mask_image], axis=3)
    # Create the cnn
    with slim.arg_scope(osmn_arg_scope()):
        net, end_points = osmn([guide_image, combined_image], model_params, is_training=False)
    probabilities = tf.nn.softmax(net)
    #net = SalSegm_vgg16({'data': combined_image})
    #probabilities_out = net.layers['softmax']
    #out_size = tf.shape(mask_image)
    #probabilities = crop_features(probabilities_out, out_size)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name]) #if '-up' not in v.name and '-cr' not in v.name])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if checkpoint_file.split('.')[1] == 'npy':
            load_model(checkpoint_file, '', 'osmn/seg')(sess)
            sess.run(interp_surgery(tf.global_variables()))
        else:
            saver.restore(sess, checkpoint_file)
            sess.run(interp_surgery(tf.global_variables()))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        dataset.reset_idx()
        for frame in range(0, dataset.get_test_size(), batch_size):
            guide_images, mask_images, images, image_paths = dataset.next_batch(batch_size, 'test')
            save_names = [ name.split('.')[0] + '.png' for name in image_paths]
            res = sess.run(probabilities, feed_dict={guide_image: guide_images, mask_image:mask_images, input_image: images})
            if model_params.adaptive_crop_testing:
                res = dataset.restore_crop(res)
            res_np = np.argmax(res.astype(np.float32), axis= 3)
            #guide_images += np.array((104, 117, 123))
            #guide_images /= 255
            for i in range(min(batch_size, dataset.get_test_size() - frame)):
                print 'Saving ' + os.path.join(result_path, save_names[i])
                if len(save_names[i].split('/')) > 1:
                    save_path = os.path.join(result_path, *(save_names[i].split('/')[:-1]))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                scipy.misc.imsave(os.path.join(result_path, save_names[i]), res_np[i].astype(np.float32))
                curr_score_name = save_names[i][:-4]
                print 'Saving ' + os.path.join(result_path, curr_score_name) + '.npy'
                np.save(os.path.join(result_path, curr_score_name), res.astype(np.float32)[i,:,:,1])

                scipy.misc.imsave(os.path.join(result_path, curr_score_name + '_mask.png'), mask_images[i,:,:,0].astype(np.uint8))
                #scipy.misc.imsave(os.path.join(result_path, curr_score_name + '_guide.png'),guide_images[i])
