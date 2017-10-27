'''
Created on 20 Oct 2017

@author: mtonnicchi
'''

class OutputManipulator(object):

    def __init__(self, max_size, use_padding=False):
        self.padding = 256
        self.max_size = max_size
        self.use_padding = use_padding

    def writeAsOutputData(self, word):
        as_data = [ord(c) for c in word]
        if self.use_padding:
            self.pad(as_data, self.padding, self.max_size)
        return as_data
        
    def readOutputData(self, output_data):
        return self.decode_word(output_data)

    def encode_word(self, word_to_encode):
        return [ord(c) for c in word_to_encode.encode('utf-8')]
        
    def decode_word(self, encoded_word):
        assert(len(encoded_word)<=self.max_size)
        data_without_padding = [number for number in encoded_word if (number != self.padding)]
        return ''.join(map(chr, data_without_padding))
    
    def decode_words(self, encoded_words):
        return '\n'.join(map(self.decode_word, encoded_words))
    
    def pad(self, given_list, padding, fit_length):
        given_list.extend([padding] * (fit_length-len(given_list)))