#!/usr/bin/env python
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

from PIL import Image
import os
import cPickle
import numpy as np

from cifar10 import Cifar10Downloader

class Cifar100FineDownloader(Cifar10Downloader):

    def getUrls(self):
        return [
            'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        ]

    def convertDataset(self, unzippedFiles):
        tarDir = unzippedFiles[0]
        datasetDir = os.path.join(self.outdir, self.getDatasetDir())
        self.mkdir(datasetDir)
        labelType = self.getLabelType()
        print "Reading and storing labels..."
        labelsOut = os.path.join(datasetDir, 'labels.txt')
        labelsFile = os.path.join(tarDir, 'meta')
        self.readAndStoreLabels(labelsFile, labelsOut, labelType+'_label_names')
        inFile = os.path.join(tarDir, 'train')
        self.__readAndStoreImages(inFile, datasetDir)
        return datasetDir

    def getUncompressedFileNames(self):
        outFiles = []
        outFiles.append(os.path.join(self.outdir, 'cifar-100-python'))
        return outFiles

    def getDatasetDir(self):
        return 'cifar-100-' + self.getLabelType() + '-train'

    def getLabelType(self):
        return 'fine'

    def __readAndStoreImages(self, inFile, outdir):
        print "Reading and storing images from %s..." % inFile
        imDim = (3, 32, 32)
        numPixels = imDim[0] * imDim[1]
        images = self.unpickle(inFile)
        labelType = self.getLabelType()
        for idx in range(len(images['data'])):
            img = np.reshape(images['data'][idx], imDim)
            label = str(images[labelType+'_labels'][idx])
            imR = Image.fromarray(img[0])
            imG = Image.fromarray(img[1])
            imB = Image.fromarray(img[2])
            img = Image.merge('RGB', (imR, imG, imB))
            direc = os.path.join(outdir, label)
            self.mkdir(direc)
            outFile = os.path.join(direc, images['filenames'][idx])
            img.save(outFile)



class Cifar100CoarseDownloader(Cifar100FineDownloader):

    def getLabelType(self):
        return 'coarse'


# This section demonstrates the usage of the above classes
if __name__ == '__main__':
    cifar = Cifar100FineDownloader('/tmp/cifar-100')
    cifar.prepareDataset()
    cifar = Cifar100CoarseDownloader('/tmp/cifar-100')
    cifar.prepareDataset()
