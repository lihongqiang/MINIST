import os
from six.moves import cPickle as pickle
import matplotlib.pyplot as plot
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def display_image(pickle_files):
    for pickle_file in pickle_files:
        print pickle_file
        if os.path.exists(pickle_file):
            try:
                with open(pickle_file, 'rb') as f:
                    letter_set = pickle.load(f)
                    mean_image = np.mean(letter_set, axis=(0))
                    print np.mean(letter_set, axis=(0)).shape

                    fig = plot.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    u = np.array(range(28))
                    x, y = np.meshgrid(u, u)
                    z = mean_image[x,y]
                    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
                    plot.show()

            except Exception as e:
                print('Unable to process data from', pickle_file, ':', e)

folder = 'notMNIST_large'
pickle_files = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.splitext(d)[-1] == '.pickle']
print pickle_files
display_image(pickle_files)

# fig = plot.figure()
# ax = fig.add_subplot(111, projection='3d')
# u = np.array(range(28))
# x, y = np.meshgrid(u, u)
# z = x ** 2 + y ** 2
# ax.plot_surface(x, y, z, rstride=4, cstride=4, cmap=cm.YlGnBu_r)
# plot.show()
