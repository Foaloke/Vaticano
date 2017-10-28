'''
Created on 14 Oct 2017

@author: mtonnicchi
'''

import random

start_token = 256
end_token = 257


class DataIteratorInputOutput(object):
    def __init__(self, vocabulary, reads_per_iteration, max_iterations, inputmanipulator, outputmanipulator):
        self.i = 0
        self.iterations = 0
        self.max_iterations = max_iterations
        self.reads_per_iteration = reads_per_iteration
        self.vocabulary = vocabulary
        random.shuffle(self.vocabulary)
        self.vocabulary_size = len(vocabulary)
        self.inputmanipulator = inputmanipulator
        self.outputmanipulator = outputmanipulator


    def __iter__(self):
        return self


    def next(self):
        if (self.vocabulary_size == 0):   
            print("Iterating on empty vocabulary!")
            raise StopIteration()
        elif(self.iterations == self.max_iterations):    
            print("Maximum iterations reached: "+str(self.iterations)+"/"+str(self.max_iterations))
            self.iterations = 0
            raise StopIteration()
        else:
            encoded = []
            decoded = []
            for _ in range(self.reads_per_iteration):
                encoded_sample, decoded_sample = self.create_sample(self.vocabulary[self.i])
                encoded.append(encoded_sample)
                decoded.append(decoded_sample)
                self.i = (self.i + 1) % self.vocabulary_size

            self.iterations = self.iterations + 1
            return encoded, decoded


    def create_sample(self, word):
        print("["+word+"]")
        return self.inputmanipulator.writeAsInputData(word), self.outputmanipulator.writeAsOutputData(word)