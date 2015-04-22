#!/usr/bin/env python
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import DatasetDownloader
from PIL import Image
import os


class MnistDownloader(DatasetDownloader.Downloader):

    def getUrls(self):
        return [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        ]

    def convertDataset(self, unzippedFiles):
        datasetDir = os.path.join(self.outdir, 'mnist-train')
        self.mkdir(datasetDir)
        self.__readDataset(unzippedFiles[0], unzippedFiles[1], datasetDir)
        return datasetDir

    def getUncompressedFileNames(self):
        outFiles = []
        for file in ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte']:
            outFiles.append(os.path.join(self.outdir, file))
        return outFiles

    def __readDataset(self, images, labels, folder):
        imfp = open(images, 'rb')
        lafp = open(labels, 'rb')
        imfp.read(4)
        lafp.read(8)
        numData = self.__readInt(imfp)
        height = self.__readInt(imfp)
        width = self.__readInt(imfp)
        print "Reading mnist data to convert it to the format for DiGiTs..."
        print "NumData=%d image=%dx%d" % (numData, height, width)
        for idx in range(0,numData):
            label = str(ord(lafp.read(1)))
            self.__storeImage(imfp, height, width, label, folder, str(idx)+'.jpg')
        lafp.close()
        imfp.close()

    def __storeImage(self, imfp, height, width, label, folder, outFile):
        direc = os.path.join(folder, label)
        self.mkdir(direc)
        outFile = os.path.join(direc, outFile)
        imStr = imfp.read(height*width)
        im = Image.frombytes('L', (height, width), imStr)
        im.save(outFile)

    def __readInt(self, fp):
        val = [ord(x) for x in fp.read(4)]
        out = (val[0] << 24) | (val[1] << 16) | (val[2] << 8) | val[3]
        return out



# This section demonstrates the usage of the above class
if __name__ == '__main__':
    mnist = MnistDownloader('/tmp/mnist')
    mnist.prepareDataset()
