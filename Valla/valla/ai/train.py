import os
import tensorflow as tf
import math
import time
import json

from collections import OrderedDict

import numpy as np

from valla.ai.model import Seq2SeqModel

from valla.ai.iterator_training import batch_size
from valla.ai.iterator_training import batch_load_per_epoch
from valla.ai.iterator_training import max_epochs
from valla.ai.iterator_training import prepare_train_batch
from valla.ai.iterator_training import max_input_size
from valla.ai.iterator_training import data_iterator_training
from valla.ai.iterator_validation import data_iterator_validation

from valla.ai.iterator_training import display_frequency_in_steps
from valla.ai.iterator_training import validation_frequency_in_steps
from valla.ai.iterator_training import model_save_frequency_in_steps

print tf.VERSION


''' Start Training '''

# Network parameters
tf.app.flags.DEFINE_string('cell_type', 'lstm', 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_string('attention_type', 'bahdanau', 'Attention mechanism: (bahdanau, luong), default: bahdanau')
tf.app.flags.DEFINE_integer('hidden_units', 1024, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('depth', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_integer('num_encoder_symbols', 258, 'Source vocabulary size')
tf.app.flags.DEFINE_integer('num_decoder_symbols', 258, 'Target vocabulary size')

tf.app.flags.DEFINE_boolean('use_residual', True, 'Use residual connection between layers')
tf.app.flags.DEFINE_boolean('attn_input_feeding', False, 'Use input feeding method in attentional decoder')
tf.app.flags.DEFINE_boolean('use_dropout', True, 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_float('dropout_rate', 0.3, 'Dropout probability for input/output/state units (0.0: no dropout)')

# Training parameters
tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
tf.app.flags.DEFINE_integer('max_epochs', max_epochs, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('max_load_batches', batch_load_per_epoch, 'Maximum # of batches to load in one epoch')
tf.app.flags.DEFINE_integer('max_seq_length', max_input_size, 'Maximum sequence length')
tf.app.flags.DEFINE_integer('display_freq', display_frequency_in_steps, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('save_freq', model_save_frequency_in_steps, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_integer('valid_data', validation_frequency_in_steps>0, 'If validation is activated')
tf.app.flags.DEFINE_integer('valid_freq', validation_frequency_in_steps, 'Evaluate model every this iteration: valid_data needed')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
tf.app.flags.DEFINE_string('model_dir', '../model_checkpoints/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'valla.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_boolean('shuffle_each_epoch', True, 'Shuffle training dataset for each epoch')
tf.app.flags.DEFINE_boolean('sort_by_length', True, 'Sort pre-fetched minibatches by their target sequence lengths')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')

FLAGS = tf.app.flags.FLAGS

def create_model(session, FLAGS):

    config = OrderedDict(sorted(FLAGS.__flags.items()))
    model = Seq2SeqModel(config, 'train')

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print 'Reloading model parameters..'
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print 'Created new model parameters..'
        session.run(tf.global_variables_initializer())
   
    return model

def train():

    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, 
        log_device_placement=FLAGS.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        # Create a log writer object
        log_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        # Create a new model or reload existing checkpoint

        #sess = tf_debug.LocalCLIDebugWrapperSession(sess) # DEBUG
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan) # DEBUG
        
        model = create_model(sess, FLAGS)

        step_time, loss = 0.0, 0.0
        words_seen, sents_seen = 0, 0
        start_time = time.time()

        # Training loop
        print 'Training (max epochs: {})..'.format(FLAGS.max_epochs)
        for _ in xrange(FLAGS.max_epochs):
            if model.global_epoch_step.eval() >= FLAGS.max_epochs:
                print 'Training is already complete.', \
                      'current epoch:{}, max epoch:{}'.format(model.global_epoch_step.eval(), FLAGS.max_epochs)
                break

            for source_seq, target_seq in data_iterator_training:
                # Get a batch from training parallel data
                source, source_len, target, target_len = prepare_train_batch(source_seq, target_seq)
                
                if source is None or target is None:
                    print 'No samples under max_seq_length ', FLAGS.max_seq_length
                    continue

                # Execute a single training step
                print(source_len)
                step_loss, summary = model.train(sess, encoder_inputs=source, encoder_inputs_length=source_len, 
                                                 decoder_inputs=target, decoder_inputs_length=target_len)

                loss += float(step_loss) / FLAGS.display_freq
                words_seen += float(np.sum(source_len+target_len))
                sents_seen += float(source.shape[0]) # batch_size

                if model.global_step.eval() % FLAGS.display_freq == 0:

                    avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")

                    time_elapsed = time.time() - start_time
                    step_time = time_elapsed / FLAGS.display_freq

                    words_per_sec = words_seen / time_elapsed
                    sents_per_sec = sents_seen / time_elapsed

                    print 'Epoch ', model.global_epoch_step.eval(), 'Step ', model.global_step.eval(), \
                          'Perplexity {0:.2f}'.format(avg_perplexity), 'Step-time ', step_time, \
                          '{0:.2f} sents/s'.format(sents_per_sec), '{0:.2f} words/s'.format(words_per_sec)

                    loss = 0
                    words_seen = 0
                    sents_seen = 0
                    start_time = time.time()

                    # Record training summary for the current batch
                    log_writer.add_summary(summary, model.global_step.eval())

                # Execute a validation step
                if model.global_step.eval() % FLAGS.valid_freq == 0:
                    print 'Validation step'
                    valid_loss = 0.0
                    valid_sents_seen = 0
                    for source_seq, target_seq in data_iterator_validation:                        
                        source, source_len, target, target_len = prepare_train_batch(source_seq, target_seq)

                        # Compute validation loss: average per word cross entropy loss
                        step_loss, summary = model.eval(sess, encoder_inputs=source, encoder_inputs_length=source_len,
                                                        decoder_inputs=target, decoder_inputs_length=target_len)
                        current_batch_size = source.shape[0]

                        valid_loss += step_loss * current_batch_size
                        valid_sents_seen += current_batch_size
                        print '  {} samples seen'.format(valid_sents_seen)

                    valid_loss = valid_loss / valid_sents_seen
                    print 'Valid perplexity: {0:.2f}'.format(math.exp(valid_loss))

                # Save the model checkpoint
                if model.global_step.eval() % FLAGS.save_freq == 0:
                    print 'Saving the model..'
                    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                    model.save(sess, checkpoint_path, global_step=model.global_step)
                    json.dump(model.config,
                              open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'wb'),
                              indent=2)

            # Increase the epoch index of the model
            model.global_epoch_step_op.eval()
            print 'Epoch {0:} DONE'.format(model.global_epoch_step.eval())
        
        print 'Saving the last model..'
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
        model.save(sess, checkpoint_path, global_step=model.global_step)
        json.dump(model.config,
                  open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'wb'),
                  indent=2)
        
    print 'Training Terminated'
