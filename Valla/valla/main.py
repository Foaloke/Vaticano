'''
Created on 4 Oct 2017

@author: mtonnicchi
'''
import valla.ai.train as tr
import valla.ai.decode as dc
import tensorflow as tf

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

tf.app.flags.DEFINE_string('model_path', '../model_checkpoints/valla.ckpt-5', 'Path to a specific model checkpoint.')

def main(_):
    tr.train()
    #dc.decode()

if __name__ == '__main__':
    tf.app.run()
