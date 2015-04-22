#!/usr/bin/env python
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

from PIL import Image
import os
import cPickle
import numpy as np

from downloader import DataDownloader

class Cifar10Downloader(DataDownloader):

    def getUrls(self):
        return [
            'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        ]

    def convertDataset(self, unzippedFiles):
        tarDir = unzippedFiles[0]
        datasetDir = os.path.join(self.outdir, 'cifar-10-train')
        self.mkdir(datasetDir)
        print "Reading and storing labels..."
        labelsOut = os.path.join(datasetDir, 'labels.txt')
        labelsFile = os.path.join(tarDir, 'batches.meta')
        self.readAndStoreLabels(labelsFile, labelsOut, 'label_names')
        for i in range(1, 6):
            idx = str(i)
            inFile = os.path.join(tarDir, 'data_batch_'+idx)
            self.__readAndStoreImages(inFile, datasetDir, idx+'_')
        return datasetDir

    def getUncompressedFileNames(self):
        outFiles = []
        outFiles.append(os.path.join(self.outdir, 'cifar-10-batches-py'))
        return outFiles

    def unpickle(self, inFile):
        ifp = open(inFile, 'rb')
        obj = cPickle.load(ifp)
        ifp.close()
        return obj

    def readAndStoreLabels(self, inFile, outFile, labelKeyName):
        labels = self.unpickle(inFile)
        ofp = open(outFile, 'w')
        for label in labels[labelKeyName]:
            ofp.write(label + '\n')
        ofp.close()

    def __readAndStoreImages(self, inFile, outdir, prefix):
        print "Reading and storing images from %s..." % inFile
        imDim = (3, 32, 32)
        numPixels = imDim[0] * imDim[1]
        images = self.unpickle(inFile)
        for idx in range(len(images['data'])):
            img = np.reshape(images['data'][idx], imDim)
            label = str(images['labels'][idx])
            imR = Image.fromarray(img[0])
            imG = Image.fromarray(img[1])
            imB = Image.fromarray(img[2])
            img = Image.merge('RGB', (imR, imG, imB))
            direc = os.path.join(outdir, label)
            self.mkdir(direc)
            outFile = os.path.join(direc, prefix + str(idx) + '.jpg')
            img.save(outFile)


# This section demonstrates the usage of the above class
if __name__ == '__main__':
    cifar = Cifar10Downloader('/tmp/cifar-10')
    cifar.prepareDataset()
