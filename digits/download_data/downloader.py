# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.

import os
import shutil
import urllib


class DataDownloader(object):
    """Base class for downloading data and setting it up for DIGITS"""

    def __init__(self, outdir, clean=False, file_extension='png'):
        """
        Arguments:
        outdir -- directory where to download and create the dataset
        if this directory doesn't exist, it will be created

        Keyword arguments:
        clean -- delete outdir first if it exists
        file_extension -- image format for output images
        """
        self.outdir = outdir
        self.mkdir(self.outdir, clean=clean)
        self.file_extension = file_extension.lower()

    def getData(self):
        """
        This is the main function that should be called by the users!
        Downloads the dataset and prepares it for DIGITS consumption
        """
        for url in self.urlList():
            self.__downloadFile(url)

        self.uncompressData()

        self.processData()
        print "Dataset directory is created successfully at '%s'" % self.outdir

    def urlList(self):
        """
        return a list of (url, output_file) tuples
        """
        raise NotImplementedError

    def uncompressData(self):
        """
        uncompress the downloaded files
        """
        raise NotImplementedError

    def processData(self):
        """
        Process the downloaded files and prepare the data for DIGITS
        """
        raise NotImplementedError

    def __downloadFile(self, url):
        """
        Downloads the url
        """
        download_path = os.path.join(self.outdir, os.path.basename(url))
        if not os.path.exists(download_path):
            print "Downloading url=%s ..." % url
            urllib.urlretrieve(url, download_path)

    def mkdir(self, d, clean=False):
        """
        Safely create a directory

        Arguments:
        d -- the directory name

        Keyword arguments:
        clean -- if True and the directory already exists, it will be deleted and recreated
        """
        if os.path.exists(d):
            if clean:
                shutil.rmtree(d)
            else:
                return
        os.mkdir(d)
