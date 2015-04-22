#!/usr/bin/env python
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os
import gzip
import PIL.Image

from downloader import DataDownloader

class MnistDownloader(DataDownloader):
    """
    See details about the MNIST dataset here:
    http://yann.lecun.com/exdb/mnist/
    """

    def urlList(self):
        return [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        ]

    def uncompressData(self):
        for zipped, unzipped in [
                ('train-images-idx3-ubyte.gz',  'train-images.bin'),
                ('train-labels-idx1-ubyte.gz',  'train-labels.bin'),
                ('t10k-images-idx3-ubyte.gz',   'test-images.bin'),
                ('t10k-labels-idx1-ubyte.gz',   'test-labels.bin'),
                ]:
            zipped_path = os.path.join(self.outdir, zipped)
            assert os.path.exists(zipped_path), 'Expected "%s" to exist' % zipped
            unzipped_path = os.path.join(self.outdir, unzipped)
            if not os.path.exists(unzipped_path):
                print "Uncompressing file=%s ..." % zipped
                with gzip.open(zipped_path) as infile, open(unzipped_path, 'wb') as outfile:
                    outfile.write(infile.read())

    def processData(self):
        self.__extract_images('train-images.bin', 'train-labels.bin', 'train')
        self.__extract_images('test-images.bin', 'test-labels.bin', 'test')

    def __extract_images(self, images_file, labels_file, output_dir):
        """
        Extract information from binary files and store them as images
        """
        with open(os.path.join(self.outdir, images_file), 'rb') as imfp, \
                open(os.path.join(self.outdir, labels_file), 'rb') as lafp:
            imfp.read(4)
            lafp.read(8)
            numData = self.__readInt(imfp)
            height = self.__readInt(imfp)
            width = self.__readInt(imfp)
            print "Extracting MNIST data from %s ..." % images_file
            print "NumData=%d image=%dx%d" % (numData, height, width)
            for idx in range(0,numData):
                label = str(ord(lafp.read(1)))
                self.__storeImage(imfp, height, width,
                        os.path.join(self.outdir, output_dir, label,
                            '%s.%s' % (idx, self.file_extension)
                            )
                        )

    def __storeImage(self, imfp, height, width, filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        imStr = imfp.read(height*width)
        im = PIL.Image.frombytes('L', (height, width), imStr)
        im.save(filename)

    def __readInt(self, fp):
        val = [ord(x) for x in fp.read(4)]
        out = (val[0] << 24) | (val[1] << 16) | (val[2] << 8) | val[3]
        return out


# This section demonstrates the usage of the above class
if __name__ == '__main__':
    mnist = MnistDownloader('/tmp/mnist')
    mnist.getData()
