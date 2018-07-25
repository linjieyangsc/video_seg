"""
One-Shot Modulater Network
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
import sys
import time
from datetime import datetime
import os
import json
import scipy.misc
from PIL import Image
from model_func import load_model, interp_surgery, osmn, osmn_masktrack, osmn_deeplab, visual_modulator
from model_func import osmn_lite, visual_modulator_lite
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import graph_util

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
    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))

    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    return final_loss


def binary_seg_summary(images, predictions, masks, gts):
    images = tf.cast(images, tf.float32)
    foreground_predictions = tf.less(0.5, predictions) 
    foreground_labels = tf.cast(foreground_predictions, tf.float32)
    batch_size = images.get_shape().as_list()[0]
    mask_binary = tf.less(0.1, masks)
    mask_binary = tf.cast(mask_binary, tf.float32)
    labels_concat = tf.concat([mask_binary, gts, foreground_labels], 3)
    results = tf.add(images * 0.5, labels_concat * 255 * 0.5)
    results = tf.cast(results, tf.uint8)
    return tf.summary.image('prediction_images', results, batch_size)

def visual_guide_summary(images):
    batch_size = images.get_shape().as_list()[0]
    results = tf.cast(images, tf.uint8)
    return tf.summary.image('visual guides', results, batch_size)

def get_model_func(base_model):
    model_dict = { 'osvos': osmn,
                    'deeplab': osmn_deeplab,
                    'masktrack': osmn_masktrack,
                    'lite': osmn_lite
                }
    if base_model in model_dict:
        return model_dict[base_model]
    else:
        raise Exception("Invalid model type!")

def train_finetune(dataset, model_params, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad=1, batch_size=1, resume_training=False, config=None, 
           use_image_summary=True, ckpt_name="osmn"):
    """Train OSMN
    Args:
    dataset: Reference to a Dataset object instance
    model_params: Model parameters
    initial_ckpt: Path to the checkpoint to initialize the whole network or visual modulator, depend on seg_ckpt
    seg_ckpt: If seg_ckpt is not None, initial_ckpt is used to initialize the visual modulator, and seg_ckpt is used to
            initialize segmentation network
    learning_rate: Value for the learning rate. It can be a number or an instance to a learning rate object.
    logs_path: Path to store the checkpoints
    max_training_iters: Number of training iterations
    save_step: A checkpoint will be created every save_steps
    display_step: Information of the training will be displayed every display_steps
    global_step: Reference to a Variable that keeps track of the training steps
    iter_mean_grad: Number of gradient computations that are average before updating the weights
    batch_size: Size of the training batch
    resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False)
    config: Reference to a Configuration object used in the creation of a Session
    use_image_summary: Boolean to use image summary during training in tensorboard
    ckpt_name: checkpoint name for saving
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

    model_func = get_model_func(model_params.base_model)
    net, end_points = model_func([guide_image, gb_image, input_image], model_params, is_training=True)


    # Define loss
    with tf.name_scope('losses'):

        main_loss = class_balanced_cross_entropy_loss(net, input_label)
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

    # Log results on training images
    if use_image_summary:
        probabilities = tf.nn.sigmoid(net)
        input_image_orig = input_image / model_params.scale_value + model_params.mean_value
        guide_image_orig = guide_image / model_params.scale_value + model_params.mean_value
        img_summary = binary_seg_summary(input_image_orig, probabilities, gb_image, input_label)
        vg_summary = visual_guide_summary(guide_image_orig)
    # Initialize variables
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        print 'Init variable'
        sess.run(init)
        tvars = tf.trainable_variables()
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # Create saver to manage checkpoints
        saver = tf.train.Saver(max_to_keep=2)

        last_ckpt_path = tf.train.latest_checkpoint(logs_path)
        if last_ckpt_path is not None and resume_training:
            # Load last checkpoint
            print('Initializing from previous checkpoint...')
            saver.restore(sess, last_ckpt_path)
            step = global_step.eval() + 1
        elif model_params.whole_model_path == '':
            print('Initializing from pre-trained imagenet model...')
            if model_params.use_visual_modulator:
                load_model(model_params.vis_mod_model_path, 'osmn/modulator')(sess)
            load_model(model_params.seg_model_path, 'osmn/seg')(sess)
            step = 1
        else:
            print('Initializing from pre-trained model...')
            load_model(model_params.whole_model_path, 'osmn')(sess)
            step = 1
        if model_params.base_model != 'lite':
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
                    curr_img_summary = sess.run([img_summary, vg_summary], feed_dict={guide_image:batch_g_image, gb_image:batch_gb_image,
                        input_image: batch_image, input_label: batch_label})
                    for s in curr_img_summary:
                        summary_writer.add_summary(s, step)
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
    model_params: Model parameters
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
    assert batch_size==1, "only allow batch size equal to 1 for testing"
    # Input data

    guide_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    gb_image = tf.placeholder(tf.float32, [batch_size, None, None, 1])
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])

    # Create model
    
    model_func = get_model_func(model_params.base_model)
    # split the model into visual modulator and other parts, visual modulator only need to run once
    if model_params.use_visual_modulator:
        if model_params.base_model =='lite':
            v_m_params = visual_modulator_lite(guide_image, model_params, is_training=False)
        else:
            v_m_params = visual_modulator(guide_image, model_params, is_training=False)
    else:
        v_m_params = None
    net, end_points = model_func([guide_image, gb_image, input_image], model_params, visual_modulator_params = v_m_params, is_training=False)
    probabilities = tf.nn.sigmoid(net)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables()]) #if '-up' not in v.name and '-cr' not in v.name])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)
        if not model_params.base_model == 'lite':
            sess.run(interp_surgery(tf.global_variables()))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        print 'start testing process'
        time_start = time.time()
        for frame in range(dataset.get_test_size()):
            guide_images, gb_images, images, save_names = dataset.next_batch(batch_size, 'test')
            # create folder for results
            if len(save_names[0].split('/')) > 1:
                save_path = os.path.join(result_path, *(save_names[0].split('/')[:-1]))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            if images is None or gb_images is None:
                # first frame of a squence
                if model_params.use_visual_modulator:
                    curr_v_m_params = sess.run(v_m_params, feed_dict={guide_image: guide_images})
                # create a black dummy image for result of the first frame, to be compatible with DAVIS eval toolkit
                scipy.misc.imsave(os.path.join(result_path, save_names[0]), np.zeros(guide_images.shape[1:3]))
            else:
                feed_dict = { gb_image:gb_images, input_image:images}
                if model_params.use_visual_modulator:
                    for v_m_param, curr_param in zip(v_m_params, curr_v_m_params):
                        feed_dict[v_m_param] = curr_param
                res_all = sess.run([probabilities], feed_dict=feed_dict)
                res = res_all[0]
                if model_params.crf_postprocessing:
                    res_np = np.zeros(res.shape[:-1])
                    res_np[0] = dataset.crf_processing(dataset.images[0], (res[0,:,:,0] > 0.5).astype(np.int32))
                else:
                    res_np = res.astype(np.float32)[:, :, :, 0] > 0.5             
                print 'Saving ' + os.path.join(result_path, save_names[0])
                scipy.misc.imsave(os.path.join(result_path, save_names[0]), res_np[0].astype(np.float32))
                curr_score_name = save_names[0][:-4]
                if model_params.save_score:
                    print 'Saving ' + os.path.join(result_path, curr_score_name) + '.npy'
                    np.save(os.path.join(result_path, curr_score_name), res.astype(np.float32)[0,:,:,0])
        time_finish = time.time()
        time_elapsed = time_finish - time_start
        print 'Total time elasped: %.3f seconds' % time_elapsed
        print 'Each frame takes %.3f seconds' % (time_elapsed / dataset.get_test_size())

def export(model_params, checkpoint_file, config=None):
    # Input data
    batch_size = 1
    im_size = model_params.im_size
    guide_image = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    gb_image = tf.placeholder(tf.float32, [batch_size, im_size[1], im_size[0], 1])
    input_image = tf.placeholder(tf.float32, [batch_size, im_size[1], im_size[0], 3])

    # Create model
    
    model_func = get_model_func(model_params.base_model)
    # split the model into visual modulator and other parts, visual modulator only need to run once
    if model_params.use_visual_modulator:
        if model_params.base_model =='lite':
            v_m_params = visual_modulator_lite(guide_image, model_params, is_training=False)
        else:
            v_m_params = visual_modulator(guide_image, model_params, is_training=False)
    else:
        v_m_params = None
    net, end_points = model_func([guide_image, gb_image, input_image], model_params, visual_modulator_params = v_m_params, is_training=False)
    probabilities = tf.nn.sigmoid(net, name = 'prob')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    rewrite_options = rewriter_config_pb2.RewriterConfig()
    rewrite_options.optimizers.append('pruning')
    rewrite_options.optimizers.append('constfold')
    rewrite_options.optimizers.append('layout')
    graph_options = tf.GraphOptions(
            rewrite_options=rewrite_options, infer_shapes=True)
    config = tf.ConfigProto(
            graph_options=graph_options,
            allow_soft_placement=True,
            )
    output_names = ['prob']
    for i, v_m_param in enumerate(v_m_params):
        visual_mod_name = 'visual_mod_params_%d' % (i+1)
        tf.identity(v_m_param, name = visual_mod_name)
        output_names.append(visual_mod_name)
    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables()]) #if '-up' not in v.name and '-cr' not in v.name])
    save_name = checkpoint_file + '.graph.pb'
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)
        if not model_params.base_model == 'lite':
            sess.run(interp_surgery(tf.global_variables()))
        output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                output_names)
        with open(save_name, 'wb') as writer:
            writer.write(output_graph_def.SerializeToString())
        model_params.output_names = output_names
        with open(save_name+'.json', 'w') as writer:
            json.dump(vars(model_params), writer)
        print 'Model saved in', save_name
