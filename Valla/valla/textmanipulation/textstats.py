'''
Created on 4 Oct 2017

@author: mtonnicchi
'''

import os
import utils.sourceutils as su

class TextStats(object):

    def __init__(self):
        pass

    def vocabulary_in_files(self, source_dir):
        vocabulary = set()
        for source_dir in source_dir.split(','):
            for source_file in os.listdir(source_dir):
                vocabulary.update(self.vocabulary_in_file(source_dir, source_file))
        return vocabulary
        
    def vocabulary_in_file(self, source_dir, source_file):
        text = su.load_source(source_dir, source_file)
        return map(lambda s : s.encode('utf-8'), text.split(' '))

