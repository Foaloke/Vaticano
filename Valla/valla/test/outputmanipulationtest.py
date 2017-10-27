'''
Created on 4 Oct 2017

@author: mtonnicchi
'''

import unittest
from valla.ai.outputmanipulator import OutputManipulator

class OutputManipulationTest(unittest.TestCase):

    def setUp(self):
        self.test_word = "test"
        self.word_as_ints = [116, 101, 115, 116, 256, 256, 256, 256, 256, 256]
        self.output_max_width = 10
        self.outputmanipulator = OutputManipulator(self.output_max_width, True)

    def testWrite(self):
        written = self.outputmanipulator.writeAsOutputData(self.test_word)
        self.assertEqual(written, self.word_as_ints)

    def testRead(self):
        read = self.outputmanipulator.readOutputData(self.word_as_ints)
        self.assertEqual(read, self.test_word)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()