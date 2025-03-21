import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

hm_epochs = 3
n_classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

"""
Let's assume n_chunks = 4 and chunk_size = 128.

If you feed a batch of 32 samples, each with 4 chunks of 128 elements, the tensor would look like this:
[
  [
    [chunk1_sample1_1, ..., chunk1_sample1_128],
    [chunk2_sample1_1, ..., chunk2_sample1_128],
    [chunk3_sample1_1, ..., chunk3_sample1_128],
    [chunk4_sample1_1, ..., chunk4_sample1_128]
  ],
  [
    [chunk1_sample2_1, ..., chunk1_sample2_128],
    [chunk2_sample2_1, ..., chunk2_sample2_128],
    [chunk3_sample2_1, ..., chunk3_sample2_128],
    [chunk4_sample2_1, ..., chunk4_sample2_128]
  ],
  ...
  [
    [chunk1_sample32_1, ..., chunk1_sample32_128],
    [chunk2_sample32_1, ..., chunk2_sample32_128],
    [chunk3_sample32_1, ..., chunk3_sample32_128],
    [chunk4_sample32_1, ..., chunk4_sample32_128]
  ]
]
In this case, the tensor has shape [32, 4, 128]
"""

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size]) # makes data consistent 
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)

