'''
Created on 4 Oct 2017

@author: mtonnicchi
'''
import valla.ai.train as tr
import valla.ai.decode as dc
from valla.Logger import mail_report
import tensorflow as tf
import time

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

tf.app.flags.DEFINE_string('model_path', '../model_checkpoints/valla.ckpt-660', 'Path to a specific model checkpoint.')

def main(_):

    start = time.time()

    tr.train()
    #dc.decode()

    end = time.time()

    mail_report.add_message("COMPLETED IN "+ str((end-start)/60) + " minutes")
    mail_report.send("Computation completed")


if __name__ == '__main__':
    tf.app.run()
