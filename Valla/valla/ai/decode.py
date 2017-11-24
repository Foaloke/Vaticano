#!/usr/bin/env python
# coding: utf-8

import gzip
import json

import tensorflow as tf
from valla.ai.iterator_decode import batch_size
from valla.ai.iterator_decode import max_decode_steps
from valla.ai.iterator_decode import data_iterator_decode
from valla.ai.iterator_decode import outputmanipulator
from valla.ai.iterator_decode import prepare_batch
from valla.ai.model import Seq2SeqModel


# Decoding parameters
tf.app.flags.DEFINE_integer('beam_width', 12, 'Beam width used in beamsearch')
tf.app.flags.DEFINE_integer('decode_batch_size', batch_size, 'Batch size used for decoding')
tf.app.flags.DEFINE_integer('max_decode_step', max_decode_steps, 'Maximum time step limit to decode')
tf.app.flags.DEFINE_boolean('write_n_best', False, 'Write n-best list (n=beam_width)')
tf.app.flags.DEFINE_string('decode_input', '../samples/input_word/test_word.png', 'Decoding input path')
tf.app.flags.DEFINE_string('decode_output', '../samples/input_word/test_word.trans', 'Decoding output path')

FLAGS = tf.app.flags.FLAGS

def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_config(FLAGS):
    
    config = unicode_to_utf8(
        json.load(open('%s.json' % FLAGS.model_path, 'rb')))
    for key, value in FLAGS.__flags.items():
        config[key] = value

    return config


def load_model(session, config):
    
    model = Seq2SeqModel(config, 'decode')
    if tf.train.checkpoint_exists(FLAGS.model_path):
        print 'Reloading model parameters..'
        model.restore(session, FLAGS.model_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(FLAGS.model_path))
    return model


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def decode():
    # Load model config
    config = load_config(FLAGS)

    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, 
        log_device_placement=FLAGS.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        # Reload existing checkpoint
        model = load_model(sess, config)
        try:
            print 'Decoding {}..'.format(FLAGS.decode_input)
            if FLAGS.write_n_best:
                fout = [fopen(("%s_%d" % (FLAGS.decode_output, k)), 'w') \
                        for k in range(FLAGS.beam_width)]
            else:
                fout = [fopen(FLAGS.decode_output, 'w')]
            
            for idx, source_seq in enumerate(data_iterator_decode):
                source, source_len = prepare_batch(source_seq)
                # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
                # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
                predicted_ids = model.predict(sess, encoder_inputs=source, 
                                              encoder_inputs_length=source_len)
                   
                # Write decoding results
                for k, f in reversed(list(enumerate(fout))):
                    for seq in predicted_ids:
                        print(str(outputmanipulator.decode_words(seq)))
                        f.write(str(outputmanipulator.decode_words(seq)) + '\n')
                    if not FLAGS.write_n_best:
                        break
                print '  {}th line decoded'.format(idx * FLAGS.decode_batch_size)
                
            print 'Decoding terminated'
        except IOError:
            pass
        finally:
            [f.close() for f in fout]


def main(_):
    decode()


if __name__ == '__main__':
    tf.app.run()