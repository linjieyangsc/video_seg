"""
Model initialization functions
"""
import tensorflow as tf
import os
slim = tf.contrib.slim

def load_model(ckpt_path, src_scope, dst_scope):
    """Initialize the network parameters from an existing model with replaced scope names
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initilaizes the network
    """
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_corresp = dict()
    
    for v in var_to_shape_map:
        corr_var = slim.get_model_variables(v.replace(src_scope, dst_scope))
        if len(corr_var) > 0 and var_to_shape_map[v] == corr_var[0].get_shape().as_list():
            vars_corresp[v] = corr_var[0]
    print 'matched variables from ckpt', ckpt_path
    print vars_corresp.keys()
    init_fn = slim.assign_from_checkpoint_fn(
            ckpt_path,
            vars_corresp)
    return init_fn

