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
num_labels = 10
def reshape(dataset, labels):
    dataset = dataset.reshape((-1, image_sizes * image_sizes)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reshape(train_dataset, train_labels)
valid_dataset, valid_labels = reshape(valid_dataset, valid_labels)
test_dataset, test_labels = reshape(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

train_subset = 20000
batch_size = 128
hide_size = 1024
graph = tf.Graph()
with graph.as_default():

    #Input data
    # tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    # tf_train_label = tf.constant(train_labels[:train_subset])
    # tf_valid_dataset = tf.constant(valid_dataset[:train_subset])
    # tf_test_dataset = tf.constant(test_dataset[:train_subset])

    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_sizes * image_sizes))
    tf_train_label = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset[:train_subset])
    tf_test_dataset = tf.constant(test_dataset[:train_subset])

    #Training data
    relu_weights = tf.Variable(tf.truncated_normal([image_sizes*image_sizes, hide_size]))
    relu_biases = tf.Variable(tf.zeros([hide_size]))

    weights = tf.Variable(tf.truncated_normal([hide_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))


    #Training computation
    relu_logits = tf.matmul(tf_train_dataset, relu_weights) + relu_biases
    relu_out = tf.nn.relu(relu_logits)

    # logits = tf.matmul(tf.nn.dropout(relu_out, 1024), weights) + biases       1.droupout
    logits = tf.matmul(relu_out, weights) + biases
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_label)) + tf.nn.l2loss(weights)    2.regularition
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_label))

    # 3.leanring rate decay
    # global_step = tf.Variable(0)  # count the number of steps taken.
    # learning_rate = tf.train.exponential_decay(0.5, global_step, 100, 0.96)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    #Predictions for the training, valid, test dataset
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(
        tf.nn.relu(tf.matmul(tf_valid_dataset, relu_weights) + relu_biases)
        , weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(
        tf.nn.relu(tf.matmul(tf_test_dataset, relu_weights) + relu_biases)
        , weights) + biases)

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / labels.shape[0]

num_steps = 3001
with tf.Session(graph=graph) as session:

    #This is a one-time operation which ensures the parameters get initialized
    tf.initialize_all_variables().run()
    print ('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_dataset = train_dataset[offset:offset+batch_size]
        batch_label = train_labels[offset:offset+batch_size]
        feed_dict = {
            tf_train_dataset:batch_dataset,
            tf_train_label:batch_label
        }
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if(step % 500 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(
                predictions, batch_label))

            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))

    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

