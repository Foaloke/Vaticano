'''
Created on 22 Oct 2017

@author: mtonnicchi
'''

import os

import dagubbio.writer.font as font
from dagubbio.writer.writer import Writer
import numpy as np
import utils.config as cfg
from valla.ai.dataiteratorinputoutput import DataIteratorInputOutput
from valla.ai.inputmanipulator import InputManipulator
from valla.ai.outputmanipulator import OutputManipulator
from valla.imagemanipulation.columniser import Columniser

from valla.ai.vocabulary import validation_vocabulary

def prepare_train_batch(seqs_x, seqs_y, maxlen=None):
    # seqs_x, seqs_y: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x <= maxlen and l_y <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    batch_size = len(seqs_x)
    
    x_lengths = np.array(lengths_x)
    y_lengths = np.array(lengths_y)

    maxlen_x = np.max(x_lengths)
    maxlen_y = np.max(y_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * inputmanipulator.padding
    y = np.ones((batch_size, maxlen_y)).astype('int32') * outputmanipulator.padding
    
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[idx, :lengths_x[idx]] = s_x
        y[idx, :lengths_y[idx]] = s_y
    return x, x_lengths, y, y_lengths


''' Load config ''' 
cfg_writer = cfg.Config('config.ini').section('WRITER')
cfg_fonts = cfg.Config('config.ini').section('FONTS_VALIDATION')
cfg_training = cfg.Config('config.ini').section('VALIDATION')

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
batch_size = int(cfg_training['batch_size'])
batch_load_per_validation_step = int(cfg_training['batch_load_per_validation_step'])

''' Init Data Iterator '''
max_word_length = max(map(len, validation_vocabulary))
max_font_size = max(map(lambda (k,v): v.size, fonts.iteritems()))
max_printed_word_width = max_word_length * max_font_size
max_printed_word_height = 2 * max_font_size
max_printed_area = max_printed_word_width * max_printed_word_height

inputmanipulator = InputManipulator(writer, max_printed_word_width, max_printed_word_height)
outputmanipulator = OutputManipulator(max_word_length*2)

data_iterator_validation = DataIteratorInputOutput(validation_vocabulary, batch_size, batch_load_per_validation_step, inputmanipulator, outputmanipulator)
