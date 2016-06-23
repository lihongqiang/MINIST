

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matlotlib backend as plotting inline in IPython


url = 'http://commondatastorage.googleapis.com/books1000/'


last_percent_reported = None
def download_progress_hook(count, blockSize, totalSize):

    global  last_percent_reported
    percent = int(count * blockSize / totalSize)

    if percent != last_percent_reported:
        if percent %5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):

    if force or not os.path.exists(filename):
        print ('Attempting to download: %s', filename)
        filename, _= urlretrieve(url+filename, filename, reporthook=download_progress_hook)
        print ('\nDownload complete!')

    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print ('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify' + filename + '. Can you get to it with a browser?'
        )
    return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)