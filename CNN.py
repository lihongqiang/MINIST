from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf


pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']

    del save
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

# (20000, 28, 28) -> (20000, 784)
# (20000, ) -> (20000, 10)

image_sizes = 28
num_channels = 1
num_labels = 10
def reshape(dataset, labels):
    dataset = dataset.reshape((-1, image_sizes, image_sizes, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reshape(train_dataset, train_labels)
valid_dataset, valid_labels = reshape(valid_dataset, valid_labels)
test_dataset, test_labels = reshape(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / labels.shape[0]

graph = tf.Graph()
depth = 16
batch_size = 16
patch_size = 5
num_hidens = 64
with graph.as_default():

    # Input data
    tf_train_dataset = tf.placeholder(tf.float32, (batch_size, image_sizes, image_sizes, num_channels))
    tf_tarin_label = tf.placeholder(tf.float32, (batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.ones([depth]))
    layer3_weights = tf.Variable(tf.truncated_normal([image_sizes // 4 * image_sizes // 4 * depth, num_hidens], stddev=0.1))
    layer3_biases = tf.Variable(tf.ones([num_hidens]))
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidens, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.ones([num_labels]))

    # Model
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], 'SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], 'SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # Training
    logits = model(train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(valid_dataset))
    test_prediction = tf.nn.softmax(model(test_dataset))

num_steps = 1001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_dataset = train_dataset[offset:offset+batch_size]
        batch_labels = train_labels[offset:offset+batch_size]
        feed_dict = {
            tf_train_dataset:batch_dataset,
            tf_tarin_label:batch_labels
        }
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0 ):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
