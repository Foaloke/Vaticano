'''
Created on 14 Oct 2017

@author: mtonnicchi
'''

import random

class DataIteratorInput(object):
    def __init__(self, vocabulary, batch_size, max_reads, inputmanipulator):
        self.i = 0
        self.reads = 0
        self.max_reads = max_reads
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        random.shuffle(self.vocabulary)
        self.vocabulary_size = len(vocabulary)
        self.inputmanipulator = inputmanipulator


    def __iter__(self):
        return self


    def next(self):
        if (self.vocabulary_size == 0):   
            print("Iterating on empty vocabulary!")
            raise StopIteration()
        elif((self.max_reads is not None) and self.reads == self.max_reads):    
            print("Maximum data read reached: "+str(self.reads)+"/"+str(self.max_reads))
            raise StopIteration()
        else:
            encoded_samples = []
            i = self.i
            for _ in range(self.batch_size):
                encoded = self.create_sample(self.vocabulary[i])
                encoded_samples.append(encoded)
                i = (i + 1) % self.vocabulary_size
                self.reads = self.reads + 1

            self.i = i
            return encoded_samples


    def create_sample(self, word):
        return self.inputmanipulator.writeAsInputData(word)

