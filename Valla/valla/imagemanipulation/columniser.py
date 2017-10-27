'''
Created on 2 Oct 2017

@author: mtonnicchi
'''

from PIL import Image
import numpy as np

class Columniser(object):

    def __init__(self):
        pass

    def columnise(self, image_path):
        img = Image.open(image_path).convert('L')
        width, _ = img.size
        img_data = list(map(self.black_white, img.getdata()))
        rows = self.chunks(img_data, width)
        columns = map(list, zip(*reversed(list(rows))))
        return columns
    
    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def black_white(self, pixel_value):
        if pixel_value > (255/2):
            return 255
        else:
            return 0
        