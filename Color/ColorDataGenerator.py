import random
import numpy as np
import colorsys

class ColorDataGenerator:

    def __init__(self, seed=None):
        self._rand = random.Random(seed)
        self._seed = seed
        

    def GetColorImages(self, numImages:int, imageSize:int, noiseLevel:float):
        return self.GetColorImagesWithLabels(numImages, imageSize, noiseLevel)[0]

    def GetColorImagesWithLabels(self, numImages:int, imageSize:int, noiseLevel:float):
        colors, labels = self._GetColors(numImages)
        images = np.empty(numImages, dtype=np.ndarray)

        for i in range(len(colors)):
            #Create a new 16x16 image with the color at index i
            image = np.ndarray((imageSize, imageSize, 3))

            for x in range(16):
                for y in range(16):
                    #Add noise to the color
                    for z in range(3):
                        image[x, y, z] = colors[i][z] + (self._rand.random() * noiseLevel - noiseLevel / 2)

            images[i] = image

        return images, labels
        

    def _GetColors(self, numColors):
        colors = np.ndarray(shape=(numColors, 3))
        labels = np.empty(numColors, dtype=float)

        for i in range(numColors):
            color = np.array([self._rand.random(), self._rand.random(), self._rand.random()])
            colors[i] = color
            labels[i] = _GetHueFromRGB(np.array(colors[i]))
        return colors, labels

def _GetHueFromRGB(rgb):
    hue, _, _ = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    return hue