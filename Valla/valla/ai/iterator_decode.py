'''
Created on 22 Oct 2017

@author: mtonnicchi
'''

import numpy as np
import os
import utils.config as cfg
import dagubbio.writer.font as font
from dagubbio.writer.writer import Writer
from valla.imagemanipulation.columniser import Columniser
from valla.ai.inputmanipulator import InputManipulator
from valla.ai.outputmanipulator import OutputManipulator
from valla.ai.dataiteratorinput import DataIteratorInput

from valla.ai.vocabulary import decoding_vocabulary

def prepare_batch(seqs_x, maxlen=None):
    # seqs_x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    batch_size = len(seqs_x)
    
    x_lengths = np.array(lengths_x)
    maxlen_x = np.max(x_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * inputmanipulator.padding
    
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
    return x, x_lengths


''' Load config ''' 
cfg_writer = cfg.Config('config.ini').section('WRITER')
cfg_fonts = cfg.Config('config.ini').section('FONTS_DECODING')
cfg_training = cfg.Config('config.ini').section('DECODING')

''' Font creation ''' 
fonts = {}
for font_name,font_data in cfg_fonts.items():
    font_data_split = font_data.split(',')
    font_path = os.path.join(cfg_writer['font_folder'], font_name+'.'+font_data_split[0])
    fonts[font_name] = font.FontInfo(font_name, font_path, int(font_data_split[1]), int(font_data_split[2]))

''' Create Writer '''
writer = Writer(fonts, int(cfg_writer['white_space_spacing']),  int(cfg_writer['padding_top']),  int(cfg_writer['spacing_correction']),  cfg.tuple(cfg_writer['warp_amplitude']),  cfg.tuple(cfg_writer['warp_period']), cfg_writer['font_output_folder'], cfg_writer['text_output_folder'])
writer.create_text_strip("test", "test")

''' Create Columniser '''
columniser = Columniser()

''' Training Config '''
max_decode_steps = int(cfg_training['max_decode_steps'])
batch_size = int(cfg_training['batch_size'])
batch_load_per_decoding_step = int(cfg_training['batch_load_per_decoding_step'])

''' Init Data Iterator '''
max_word_length = max(map(len, decoding_vocabulary))
max_font_size = max(map(lambda (k,v): v.size, fonts.iteritems()))
max_printed_word_width = max_word_length * max_font_size
max_printed_word_height = 2 * max_font_size
max_printed_area = max_printed_word_width * max_printed_word_height

inputmanipulator = InputManipulator(writer, max_printed_word_width, max_printed_word_height)
outputmanipulator = OutputManipulator(max_word_length*2)

data_iterator_instance = DataIteratorInput(decoding_vocabulary, batch_size, batch_load_per_decoding_step, inputmanipulator)
