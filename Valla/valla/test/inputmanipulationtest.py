'''
Created on 4 Oct 2017

@author: mtonnicchi
'''

import os
from shutil import copyfile
import unittest
from mock import Mock
from valla.ai.inputmanipulator import InputManipulator

from PIL import Image
class InputManipulationTest(unittest.TestCase):

    def setUp(self):

        #self.image_array = \
        #    [1, 0, 1, 1, 0, 0, 0,
        #     0, 1, 1, 0, 1, 0, 0,
        #     0, 1, 1, 0, 1, 0, 1]
        self.image_array = [176, 209, 168]
        self.image_max_width = 3 
        self.image_max_height = 7
        self.destination_test_image = "../resources/test/images/generated_by_inputmanipulationtest.png"
        self.destination_test_image_expected = "../resources/test/images/generated_by_inputmanipulationtest_expected.png"
        self.reference_test_image_copy = "../resources/test/images/test_word_copy_that_will_be_deleted.png"
        copyfile(self.destination_test_image_expected, self.reference_test_image_copy)
        
        self.writer = Mock()
        self.writer.create_text_strip = Mock(return_value=self.reference_test_image_copy)

        self.inputmanipulator = InputManipulator(self.writer, self.image_max_width, self.image_max_height)

    def testWrite(self):
        written = self.inputmanipulator.writeAsInputData("ignored string as method return is mocked")
        self.assertEqual(written, self.image_array)

    def testRead(self):
        self.inputmanipulator.readInputData(self.image_array, self.destination_test_image)
        self.assertEqual(open(self.destination_test_image,"rb").read(), open(self.destination_test_image_expected,"rb").read())
        os.remove(self.destination_test_image)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()