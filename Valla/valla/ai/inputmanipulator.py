'''
Created on 20 Oct 2017

@author: mtonnicchi
'''

import os
import numpy as np
import math
from PIL import Image

class InputManipulator(object):

    def __init__(self, writer, x_max_size, y_max_size, use_padding_x=False, use_padding_y=True):
        self.white = 0
        self.black = 1
        self.binary_encoding_power = 8 
        self.pixel_padding = self.white
        self.padding = 256
        self.writer = writer
        self.x_max_size = x_max_size
        self.y_max_size = y_max_size
        self.use_padding_x = use_padding_x
        self.use_padding_y = use_padding_y
    
    def writeAsInputData(self, word):
        generated_file = self.writer.create_text_strip("inputmanipulator_tmp", word)
        columns = self.columnise(generated_file)
        os.remove(generated_file)
        
        if self.use_padding_y:
            map( lambda c: self.pad(c, self.pixel_padding, self.y_max_size), columns )
        
        symbols = self.make_symbols([pixel for column in columns for pixel in column])
        
        if self.use_padding_x:
            self.pad(symbols, self.padding, self.x_max_size)
        return symbols
        
    def readInputData(self, input_data, destination_path):
        assert(len(input_data) <= math.ceil((float)(self.y_max_size*self.x_max_size)/self.binary_encoding_power))
        binary_sequence = self.make_binary_sequence(input_data)
        columns = self.chunks(binary_sequence, self.y_max_size)
        rows = map(list, zip(*list(columns)))
        img_data = [pixel for row in rows for pixel in row]
        image_out = Image.new('L', (self.x_max_size,self.y_max_size))
        image_out.putdata(map(self.from_black_and_white, img_data))
        image_out.save(destination_path)
    
    def columnise(self, image_path):
        img = Image.open(image_path).convert('L')
        width, _ = img.size
        img_data = list(map(self.as_black_and_white, img.getdata()))
        rows = self.chunks(img_data, width)
        columns = map(list, zip(*list(rows)))
        return columns

    def as_black_and_white(self, pixel_value):
        if pixel_value > (255/2):
            return self.white
        else:
            return self.black

    def from_black_and_white(self, value):
        if value == self.white:
            return 255
        else:
            return 0

    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    
    def pad(self, given_list, padding, fit_length):
        given_list.extend([padding] * (fit_length-len(given_list)))
        
    def make_symbols(self, binary_digits):
        while len(binary_digits) % self.binary_encoding_power != 0:
            binary_digits.append(self.pixel_padding)
        reshaped = np.reshape(binary_digits, (len(binary_digits)/self.binary_encoding_power,self.binary_encoding_power))
        return map(self.make_symbol, reshaped)

    def make_symbol(self, binary_digit_sequence):
        return int(''.join(map(str, binary_digit_sequence)), 2)
    
    def make_binary_sequence(self, symbols):
        binary_sequence = map(int, ''.join(map(lambda s: str(bin(s))[2:], symbols)))
        padding = len(binary_sequence) % self.y_max_size
        return binary_sequence[:-padding]