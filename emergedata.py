import numpy as np
import os
from six.moves import cPickle as pickle

image_size = 28

def make_arrays(nb_row, img_size):
    if nb_row:
        dataset = np.ndarray(shape=(nb_row, img_size, img_size), dtype=np.float32)
        labelset = np.ndarray(shape=(nb_row), dtype=np.int32)
    else:
        dataset, labelset = None, None
    return dataset, labelset



def merge_datasets(pickle_files, train_sizes, valid_sizes=0):

    num_classes = len(pickle_files)
    valid_dataset, valid_label = make_arrays(valid_sizes*num_classes, image_size)
    train_dataset, train_label = make_arrays(train_sizes*num_classes, image_size)


    train_start, train_end = 0, train_sizes
    valid_start, valid_end = 0, valid_sizes
    data_start, data_end = valid_sizes, valid_sizes + train_sizes
    
    train_per_class = train_sizes
    valid_per_class = valid_sizes

    print 'valid_per_class = ' + str(valid_per_class) + ' train_per_class = ' + str(train_per_class)

    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                print letter_set.shape
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    print valid_start, valid_end
                    valid_letter = letter_set[:valid_per_class, :, :]
                    valid_dataset[valid_start:valid_end, :, :] = valid_letter
                    valid_label[valid_start:valid_end] = label
                    valid_start += valid_per_class
                    valid_end += valid_per_class

                train_dataset[train_start:train_end, :, :] = letter_set[data_start:data_end, :, :]
                train_label[train_start:train_end] = label
                train_start += train_per_class
                train_end += train_per_class

        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise e

    return valid_dataset, valid_label, train_dataset, train_label

train_size = 20000
valid_size = 1000
test_size = 1000
train_folder = 'notMNIST_large'
train_datasets = [os.path.join(train_folder, d) for d in os.listdir(train_folder) if os.path.splitext(d)[-1] == '.pickle']
test_folder = 'notMNIST_small'
test_datasets = [os.path.join(train_folder, d) for d in os.listdir(train_folder) if os.path.splitext(d)[-1] == '.pickle']

valid_dataset, valid_labels, \
train_dataset, train_labels \
    = merge_datasets(train_datasets, train_size, valid_size)
_, _, \
test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


def randomize(dataset, label):
    permutation = np.random.permutation(label.shape[0])
    shuffle_dataset = dataset[permutation, :, :]
    shuffle_label = label[permutation]
    return shuffle_dataset, shuffle_label

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

pickle_file = 'notMNIST.pickle'
try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to ', pickle_file, ':' , e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)