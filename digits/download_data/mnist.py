# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.

import gzip
import os
import struct

import numpy as np
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

    def __extract_images(self, images_file, labels_file, phase):
        """
        Extract information from binary files and store them as images
        """
        labels = self.__readLabels(os.path.join(self.outdir, labels_file))
        images = self.__readImages(os.path.join(self.outdir, images_file))
        assert len(labels) == len(images), '%d != %d' % (len(labels), len(images))

        output_dir = os.path.join(self.outdir, phase)
        self.mkdir(output_dir, clean=True)
        with open(os.path.join(output_dir, 'labels.txt'), 'w') as outfile:
            for label in xrange(10):
                outfile.write('%s\n' % label)
        with open(os.path.join(output_dir, '%s.txt' % phase), 'w') as outfile:
            for index, image in enumerate(images):
                dirname = os.path.join(output_dir, labels[index])
                self.mkdir(dirname)
                filename = os.path.join(dirname, '%05d.%s' % (index, self.file_extension))
                image.save(filename)
                outfile.write('%s %s\n' % (filename, labels[index]))

    def __readLabels(self, filename):
        """
        Returns a list of ints
        """
        print 'Reading labels from %s ...' % filename
        labels = []
        with open(filename, 'rb') as infile:
            infile.read(4)  # ignore magic number
            count = struct.unpack('>i', infile.read(4))[0]
            data = infile.read(count)
            for byte in data:
                label = struct.unpack('>B', byte)[0]
                labels.append(str(label))
        return labels

    def __readImages(self, filename):
        """
        Returns a list of PIL.Image objects
        """
        print 'Reading images from %s ...' % filename
        images = []
        with open(filename, 'rb') as infile:
            infile.read(4)  # ignore magic number
            count = struct.unpack('>i', infile.read(4))[0]
            rows = struct.unpack('>i', infile.read(4))[0]
            columns = struct.unpack('>i', infile.read(4))[0]

            for i in xrange(count):
                data = infile.read(rows * columns)
                image = np.fromstring(data, dtype=np.uint8)
                image = image.reshape((rows, columns))
                image = 255 - image  # now black digit on white background
                images.append(PIL.Image.fromarray(image))
        return images
