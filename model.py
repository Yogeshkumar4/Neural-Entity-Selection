import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.layers.core import Dense
import random
np.set_printoptions(threshold=np.nan)

learning_rate = 0.001
batch_size = 128
display_step = 5
epochs = 100
filename = 'glove.840B.300d.txt'

def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd

vocab,embd = loadGloVe(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
max_document_length = 50

# Modeling Embedding layer
W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),trainable=False, name="W")
embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = W.assign(embedding_placeholder)

def last_relevant(output, length):
  batch_size = tf.shape(output)[0]
  max_length = tf.shape(output)[1]
  out_size = int(output.get_shape()[2])
  index = tf.range(0, batch_size)*max_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant

# Modeling the encoder
num_encoder_layers = 2
encoder_hidden_units = 256

encoder_inputs = tf.placeholder(dtype=tf.float32, shape=(None, max_document_length))
encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,))

encoder_cell = MultiRNNCell([LSTMCell(encoder_hidden_units) for i in range(num_encoder_layers)])
encoder_inputs_embedded = tf.nn.embedding_lookup(W, encoder_inputs)
input_layer = Dense(encoder_hidden_units, dtype=tf.float32)
encoder_inputs_embedded_l = input_layer(encoder_inputs_embedded)

encoder_outputs, _ = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_inputs_embedded_l,sequence_length=encoder_inputs_length, dtype=tf.float32)
encoder_means = tf.reduce_mean(encoder_outputs, 1)
last_encoder = last_relevant(encoder_outputs, encoder_inputs_length)
# Modeling the decoder
n_hidden_1 = 256 
n_hidden_2 = 256 
n_input = encoder_hidden_units*3
n_classes = 1 

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

logits = multilayer_perceptron(X)

loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()

def nextBatch(j, input_l, target_l):
    c = list(zip(input_l[j*batch_size:(j+1)*batch_size], target_l[j*batch_size:(j+1)*batch_size]))
    random.shuffle(c)
    return zip(*c)

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
pretrain = vocab_processor.fit(vocab)

# f = open("./ner_file.txt","r").readlines()
# answers = []
# target = []
# for line in f:
#     sent = line.split("@~@")
#     answers.append(sent[0])
#     target.append([int(sent[1].rstrip())])
# answers = ["this is my dream", "hello world"]
answers = []
answers.append("the history of pubs can be traced back to roman taverns , through the anglo-saxon alehouse to the development of the modern tied house system in the 19th century .")
answers.append("most pubs offer a range of beers , wines , spirits , and soft drinks and snacks .")
answers.append("the owner , tenant or manager -lrb- licensee -rrb- of a pub is properly known as the `` pub landlord '' .")
x = np.array(list(vocab_processor.transform(answers)))
lens = [len(elem.split(" ")) for elem in answers]
print(x)
with tf.Session() as sess:
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
    sess.run(init)
    encoder_out, means, last_rel, check = sess.run([encoder_outputs, encoder_means, last_encoder, encoder_inputs_embedded], feed_dict={encoder_inputs:x,encoder_inputs_length:lens})
    print(encoder_out)
    print("--------------------")
    print(means)
    print("--------------------")
    print(last_rel)
    print("--------------------")
    print(check)
    # myout = sess.run(output, feed_dict={input_data: x})
    # embedded_sents = np.reshape(myout, (-1, n_input))
    # numBatches = len(embedded_sents)/batch_size
    # embedded_sents = embedded_sents[:numBatches*batch_size]
    # target = target[:numBatches*batch_size]
    # sess.run(init)
    # for i in range(epochs):
    #     training_loss = 0
    #     for j in range(numBatches):
    #         batchx, batchy = nextBatch(j, embedded_sents, target)
    #         _, curr_loss = sess.run([train_op, loss_op], feed_dict={X:batchx, Y:batchy})
    #         training_loss += curr_loss
    #     if i%display_step == 0:
    #         print("Epoch:{0}\tTraining Loss:{1}".format(i, training_loss/len(embedded_sents)))    
