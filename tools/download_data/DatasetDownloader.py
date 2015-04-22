# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os
import gzip
from subprocess import call
import urllib


class Downloader:
    """Base class for downloading and setting up a DiGiTS dataset"""

    def __init__(self, outdir):
        """
        Arguments:
        outdir -- directory where to download and create the dataset
        if this directory doesn't exist, it will be created
        """
        self.outdir = outdir
        self.mkdir(self.outdir)

    def prepareDataset(self):
        """
        This is the main function that should be called by the users!
        Downloads the dataset and prepares it for DiGiTS consumption
        """
        urls = self.getUrls()
        files = self.__downloadFiles(urls)
        self.__uncompressFiles(files)
        unzippedFiles = self.getUncompressedFileNames()
        datasetDir = self.convertDataset(unzippedFiles)
        print "Dataset directory is created successfully at '%s'" % datasetDir
        return datasetDir

    def getUrls(self):
        """
        return a list of all urls to be downloaded
        """
        raise NotImplementedError()

    def convertDataset(self, unzippedFiles):
        """
        Arguments:
        unzippedFiles -- list of all uncompressed files and prepare the dataset
        returns the outdir containing the training dataset as expected by DiGiTS
        """
        raise NotImplementedError()

    def getUncompressedFileNames(self):
        """
        In some datasets, there is no relation between the uncompressed file names
        and their compressed counter parts. Thus, it is better to ask each dataset
        class to explicitly create the filepaths for the uncompressed files!
        """
        raise NotImplementedError()

    def __downloadFiles(self, urls):
        files = []
        for url in urls:
            print "Downloading url=%s ..." % url
            tmp = os.path.join(self.outdir, os.path.basename(url))
            files.append(tmp)
            if not os.path.exists(tmp):
                urllib.urlretrieve(url, tmp)
        return files

    def __uncompressFiles(self, files):
        for file in files:
            print "Uncompressing file=%s ..." % file
            if self.__tarGzipped(file):
                self.__untar(file, '-xzf')
            elif self.__gzipped(file):
                self.__gunzip(file)
            else:
                raise Exception('Unsupported compression format!')

    def __tarGzipped(self, f):
        if len(f) < 7:
            return 0
        if f[-7:] == '.tar.gz':
            return True
        return False

    def __untar(self, f, option):
        dir = os.path.dirname(f)
        call(['tar', option, f, '-C', dir])
        out = f.replace('.tar.gz', '')
        return out

    def __gzipped(self, f):
        if len(f) < 3:
            return 0
        if f[-3:] == '.gz':
            return True
        return False

    def __gunzip(self, f):
        outFile = f.replace('.gz', '')
        fp = gzip.open(f, 'rb')
        outFp = open(outFile, 'wb')
        outFp.write(fp.read())
        outFp.close()
        fp.close()
        return outFile

    def mkdir(self, d):
        if os.path.exists(d):
            return
        os.mkdir(d)
