import tensorflow as tf
import os
slim = tf.contrib.slim

def instance_normalization(inputs,reuse=None, variables_collections=None,output_collections=None,
        use_biases=True, trainable=True, scope=None):
    with tf.variable_scope(scope, 'InstanceNorm', [inputs],
                         reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
          raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        if inputs_rank != 4:
          raise ValueError('Inputs %s is not a 4D tensor.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        axis = [1, 2]
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
          raise ValueError('Inputs %s has undefined last dimension %s.' % (
              inputs.name, params_shape))

        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        #var_collections = slim.utils.get_variable_collections(
        #        variables_collections, name)
        dtype = inputs.dtype.base_dtype
        shape = tf.TensorShape([1, 1, 1]).concatenate(params_shape)
        beta = slim.model_variable('beta', shape=shape, dtype=dtype,
                initializer=tf.zeros_initializer(), collections=None,
                trainable=use_biases)
        gamma = slim.model_variable('gamma', shape=shape, dtype=dtype,
                initializer=tf.ones_initializer(), collections=None,
                trainable=trainable)
        if use_biases:
            print 'use biases'
        else:
            print 'not use biases'
        outputs = inputs * gamma + beta
        return slim.utils.collect_named_outputs(output_collections,
                                                sc.original_name_scope,
                                                outputs)
def conditional_normalization(inputs, gamma, reuse=None, variable_collections=None,
                        output_collections=None, trainable=True, scope=None):
    with tf.variable_scope(scope, 'ConditionalNorm', [inputs, gamma],
                        reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims

        if inputs_rank is None:
          raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        if inputs_rank != 4:
          raise ValueError('Inputs %s is not a 4D tensor.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        axis = [1, 2]
        params_shape = inputs_shape[-1:]
        gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
        if not params_shape.is_fully_defined():
          raise ValueError('Inputs %s has undefined last dimension %s.' % (
              inputs.name, params_shape))
        return inputs * gamma
