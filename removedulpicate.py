from six.moves import cPickle as pickle
import numpy as np
import os

pickle_file = 'notMNIST.pickle'
new_pickle_file = 'new_notMNIST.pickle'

def remove_dulpicate(pickle_file):
    if os.path.exists(pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                data_dict = pickle.load(f)
                new_dict = {}
                for index, data in data_dict.items():
                    new_data = np.array(list(set(data)))
                    new_dict[index] = new_data
                try:
                    f = open(new_pickle_file, 'wb')
                    pickle.dump(new_data, f, pickle.HIGHEST_PROTOCOL)
                    statinfo = os.stat(new_pickle_file)
                    print statinfo.st_size
                except Exception as e:
                    print ('Could not create file ', new_pickle_file, ':', e)
                    raise
        except Exception as ee:
            print ('Could not opne file ', pickle_file, ':', ee)
            raise

remove_dulpicate(pickle_file)

import sklearn.linear_model

