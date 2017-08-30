# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.

import cPickle
import os
import tarfile

import PIL.Image

from downloader import DataDownloader


class Cifar10Downloader(DataDownloader):
    """
    See details about the CIFAR10 dataset here:
    http://www.cs.toronto.edu/~kriz/cifar.html
    """

    def urlList(self):
        return [
            'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        ]

    def uncompressData(self):
        filename = 'cifar-10-python.tar.gz'
        filepath = os.path.join(self.outdir, filename)
        assert os.path.exists(filepath), 'Expected "%s" to exist' % filename

        if not os.path.exists(os.path.join(self.outdir, 'cifar-10-batches-py')):
            print "Uncompressing file=%s ..." % filename
            with tarfile.open(filepath) as tf:
                tf.extractall(self.outdir)

    def processData(self):
        label_filename = 'batches.meta'
        label_filepath = os.path.join(self.outdir, 'cifar-10-batches-py', label_filename)
        with open(label_filepath, 'rb') as infile:
            pickleObj = cPickle.load(infile)
            label_names = pickleObj['label_names']

        for phase in 'train', 'test':
            dirname = os.path.join(self.outdir, phase)
            self.mkdir(dirname, clean=True)
            with open(os.path.join(dirname, 'labels.txt'), 'w') as outfile:
                for name in label_names:
                    outfile.write('%s\n' % name)

        for filename, phase in [
                ('data_batch_1', 'train'),
                ('data_batch_2', 'train'),
                ('data_batch_3', 'train'),
                ('data_batch_4', 'train'),
                ('data_batch_5', 'train'),
                ('test_batch', 'test'),
        ]:
            filepath = os.path.join(self.outdir, 'cifar-10-batches-py', filename)
            assert os.path.exists(filepath), 'Expected "%s" to exist' % filename

            self.__extractData(filepath, phase, label_names)

    def __extractData(self, input_file, phase, label_names):
        """
        Read a pickle file at input_file and output images

        Arguments:
        input_file -- the input pickle file
        phase -- train or test
        label_names -- a list of strings
        """
        print 'Extracting images file=%s ...' % input_file

        # Read the pickle file
        with open(input_file, 'rb') as infile:
            pickleObj = cPickle.load(infile)
            # print 'Batch -', pickleObj['batch_label']
            data = pickleObj['data']
            assert data.shape == (10000, 3072), 'Expected data.shape to be (10000, 3072), not %s' % (data.shape,)
            count = data.shape[0]
            labels = pickleObj['labels']
            assert len(labels) == count, 'Expected len(labels) to be %d, not %d' % (count, len(labels))
            filenames = pickleObj['filenames']
            assert len(filenames) == count, 'Expected len(filenames) to be %d, not %d' % (count, len(filenames))

        data = data.reshape((10000, 3, 32, 32))
        data = data.transpose((0, 2, 3, 1))

        output_dir = os.path.join(self.outdir, phase)
        self.mkdir(output_dir)
        with open(os.path.join(output_dir, '%s.txt' % phase), 'a') as outfile:
            for index, image in enumerate(data):
                # Create the directory
                dirname = os.path.join(output_dir, label_names[labels[index]])
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                # Get the filename
                filename = filenames[index]
                ext = os.path.splitext(filename)[1][1:].lower()
                if ext != self.file_extension:
                    filename = '%s.%s' % (os.path.splitext(filename)[0], self.file_extension)
                filename = os.path.join(dirname, filename)

                # Save the image
                PIL.Image.fromarray(image).save(filename)
                outfile.write('%s %s\n' % (filename, labels[index]))
