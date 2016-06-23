import os
import sys
import tarfile

num_class = 10

def maybe_extract(filename, force=False):

    root = os.path.splitext(os.path.split(filename)[0])[0]

    if force or not os.path.isdir(filename):
        print ('Exacting data for %s. This may takes a while. Please wait.', filename)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
        data_floders = [os.path.join(root, d) for d in sorted(os.listdir(root))
                        if os.path.isdir(os.path.join(root, d))]
        if len(data_floders) != num_class:
            raise Exception(
                'Expected %d folders, one per class. Found %d instead.', num_class, len(data_floders)
            )
        print (data_floders)
        return data_floders
    else:
        print ('%s already present - Skipping extraction of %s.' % (root, filename))

train_folders = maybe_extract('notMNIST_large.tar.gz')
test_folders = maybe_extract('notMNIST_small.tar.gz')
