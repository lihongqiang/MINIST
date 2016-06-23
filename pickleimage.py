import os
from scipy import ndimage
from six.moves import cPickle as pickle
import numpy as np

image_size = 28 # Pixel width and height.
pixel_depth = 255.0 # Number of levels per pixel.


def load_letter(folder, min_num_image):
    print folder
    imagefiles = os.listdir(folder)
    dataset = np.ndarray(shape=(len(imagefiles), image_size, image_size), dtype=np.float32)
    num_images = 0
    for image in imagefiles:
        image_file = os.path.join(folder, image)
        try:
            # -0.5~0.5
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2.0) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images += 1
        except Exception as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:min_num_image, :, :]
    if num_images < min_num_image:
        raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_image))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset



def maybe_pickle(data_folders, min_num_image_per_class, force=False):

    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print ('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_image_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
    return dataset_names

root = 'notMNIST_large'
data_folders = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
print data_folders
maybe_pickle(data_folders, 45000)

root = 'notMNIST_small'
data_folders = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
print data_folders
maybe_pickle(data_folders, 1800)

