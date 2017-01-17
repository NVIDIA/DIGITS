# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.

import cPickle
import os
import tarfile

import PIL.Image

from downloader import DataDownloader


class Cifar100Downloader(DataDownloader):
    """
    See details about the CIFAR100 dataset here:
    http://www.cs.toronto.edu/~kriz/cifar.html
    """

    def urlList(self):
        return [
            'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
        ]

    def uncompressData(self):
        filename = 'cifar-100-python.tar.gz'
        filepath = os.path.join(self.outdir, filename)
        assert os.path.exists(filepath), 'Expected "%s" to exist' % filename

        if not os.path.exists(os.path.join(self.outdir, 'cifar-100-python')):
            print "Uncompressing file=%s ..." % filename
            with tarfile.open(filepath) as tf:
                tf.extractall(self.outdir)

    def processData(self):
        label_filename = 'meta'
        label_filepath = os.path.join(self.outdir, 'cifar-100-python', label_filename)
        with open(label_filepath, 'rb') as infile:
            pickleObj = cPickle.load(infile)
            fine_label_names = pickleObj['fine_label_names']
            coarse_label_names = pickleObj['coarse_label_names']

        for level, label_names in [
                ('fine', fine_label_names),
                ('coarse', coarse_label_names),
        ]:
            dirname = os.path.join(self.outdir, level)
            self.mkdir(dirname, clean=True)
            with open(os.path.join(dirname, 'labels.txt'), 'w') as outfile:
                for name in label_names:
                    outfile.write('%s\n' % name)

        for filename, phase in [
                ('train', 'train'),
                ('test', 'test'),
        ]:
            filepath = os.path.join(self.outdir, 'cifar-100-python', filename)
            assert os.path.exists(filepath), 'Expected "%s" to exist' % filename

            self.__extractData(filepath, phase, fine_label_names, coarse_label_names)

    def __extractData(self, input_file, phase, fine_label_names, coarse_label_names):
        """
        Read a pickle file at input_file and output as images

        Arguments:
        input_file -- a pickle file
        phase -- train or test
        fine_label_names -- mapping from fine_labels to strings
        coarse_label_names -- mapping from coarse_labels to strings
        """
        print 'Extracting images file=%s ...' % input_file

        # Read the pickle file
        with open(input_file, 'rb') as infile:
            pickleObj = cPickle.load(infile)
            # print 'Batch -', pickleObj['batch_label']
            data = pickleObj['data']
            assert data.shape[1] == 3072, 'Unexpected data.shape %s' % (data.shape,)
            count = data.shape[0]
            fine_labels = pickleObj['fine_labels']
            assert len(fine_labels) == count, 'Expected len(fine_labels) to be %d, not %d' % (count, len(fine_labels))
            coarse_labels = pickleObj['coarse_labels']
            assert len(coarse_labels) == count, 'Expected len(coarse_labels) to be %d, not %d' % (
                count, len(coarse_labels))
            filenames = pickleObj['filenames']
            assert len(filenames) == count, 'Expected len(filenames) to be %d, not %d' % (count, len(filenames))

        data = data.reshape((count, 3, 32, 32))
        data = data.transpose((0, 2, 3, 1))

        fine_to_coarse = {}  # mapping of fine labels to coarse labels

        fine_dirname = os.path.join(self.outdir, 'fine', phase)
        os.makedirs(fine_dirname)
        coarse_dirname = os.path.join(self.outdir, 'coarse', phase)
        os.makedirs(coarse_dirname)
        with open(os.path.join(self.outdir, 'fine', '%s.txt' % phase), 'w') as fine_textfile, \
                open(os.path.join(self.outdir, 'coarse', '%s.txt' % phase), 'w') as coarse_textfile:
            for index, image in enumerate(data):
                # Create the directory
                fine_label = fine_label_names[fine_labels[index]]
                dirname = os.path.join(fine_dirname, fine_label)
                self.mkdir(dirname)

                # Get the filename
                filename = filenames[index]
                ext = os.path.splitext(filename)[1][1:].lower()
                if ext != self.file_extension:
                    filename = '%s.%s' % (os.path.splitext(filename)[0], self.file_extension)
                filename = os.path.join(dirname, filename)

                # Save the image
                PIL.Image.fromarray(image).save(filename)
                fine_textfile.write('%s %s\n' % (filename, fine_labels[index]))
                coarse_textfile.write('%s %s\n' % (filename, coarse_labels[index]))

                if fine_label not in fine_to_coarse:
                    fine_to_coarse[fine_label] = coarse_label_names[coarse_labels[index]]

        # Create the coarse dataset with symlinks
        for fine, coarse in fine_to_coarse.iteritems():
            self.mkdir(os.path.join(coarse_dirname, coarse))
            os.symlink(
                # Create relative symlinks for portability
                os.path.join('..', '..', '..', 'fine', phase, fine),
                os.path.join(coarse_dirname, coarse, fine)
            )
