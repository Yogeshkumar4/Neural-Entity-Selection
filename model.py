import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.layers.core import Dense
import random
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')
np.set_printoptions(threshold=np.nan)

learning_rate = 0.001
batch_size = 128
display_step = 5
epochs = 100
filename = './glove.840B.300d.txt'
train = False

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
max_document_length = 200

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

encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, max_document_length))
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
probs = tf.nn.sigmoid(logits)
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()

def nextBatch(j, input_l, target_l, lens_l, ner_len_l):
    c = list(zip(input_l[j*batch_size:(j+1)*batch_size], target_l[j*batch_size:(j+1)*batch_size], lens_l[j*batch_size:(j+1)*batch_size], ner_len_l[j*batch_size:(j+1)*batch_size]))
    random.shuffle(c)
    return zip(*c)

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
pretrain = vocab_processor.fit(vocab)


# Reading the dataset
if train:
    f = open("./NES_Dataset.txt","r").readlines()
else:
    f = open("./test_set.txt","r").readlines()
answers = []
ner_length = []
target = []
for line in f:
    sent = line.split("@~@")
    position = sent[1].split(",")
    ner_start = int(position[0])
    ner_end = int(position[1]) + 1
    if (ner_end > max_document_length):
        continue 
    answers.append(sent[0])
    ner_length.append(range(ner_start,ner_end))
    if train:
        target.append([int(sent[2].rstrip())])
    else:
        target.append(0)
x = np.array(list(vocab_processor.transform(answers)))
my_ans = x.tolist()
lens = [min(len(elem.split(" ")),max_document_length) for elem in answers]
max_score = {}
my_ner = {}
with tf.Session() as sess:
    sess.run(init)
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
    numBatches = len(x)/batch_size
    x = x[:numBatches*batch_size]
    lens = lens[:numBatches*batch_size]
    ner_length = ner_length[:numBatches*batch_size]
    target = target[:numBatches*batch_size]
    if train:
        saver = tf.train.Saver()
        for i in range(epochs):
            training_loss = 0
            for j in range(numBatches):
                batchx, batchy, batchlens, batchner_length = nextBatch(j, x, target, lens, ner_length)
                enc_outputs, enc_means, last_enc = sess.run([encoder_outputs, encoder_means, last_encoder], feed_dict={encoder_inputs:batchx,encoder_inputs_length:batchlens})
                till_now = np.concatenate((last_enc, enc_means), axis=1)
                to_calc = np.zeros(shape=(len(batchx),encoder_hidden_units))
                for k in range(len(batchx)):
                    to_calc[k] = np.mean(enc_outputs[k][np.ix_(batchner_length[k],range(encoder_hidden_units))],0)
                mlp_input = np.concatenate((till_now, to_calc), axis=1)
                _, curr_loss = sess.run([train_op, loss_op], feed_dict={X:mlp_input, Y:batchy})
                training_loss += curr_loss
            if i%display_step == 0:
                print("Epoch:{0}\tTraining Loss:{1}".format(i, training_loss/len(x)))   
        saver.save(sess, 'my-model') 
    else:
        saver = tf.train.import_meta_graph('my-model.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))  
        for i in range(numBatches):
            batchx, batchy, batchlens, batchner_length = nextBatch(i, x, target, lens, ner_length)
            enc_outputs, enc_means, last_enc = sess.run([encoder_outputs, encoder_means, last_encoder], feed_dict={encoder_inputs:batchx,encoder_inputs_length:batchlens})
            till_now = np.concatenate((last_enc, enc_means), axis=1)
            to_calc = np.zeros(shape=(len(batchx),encoder_hidden_units))
            for k in range(len(batchx)):
                to_calc[k] = np.mean(enc_outputs[k][np.ix_(batchner_length[k],range(encoder_hidden_units))],0)
            mlp_input = np.concatenate((till_now, to_calc), axis=1)
            predictions = sess.run([probs], feed_dict={X:mlp_input})
            for elem in range(len(batchx)):
                item = answers[my_ans.index(batchx[elem].tolist())]
                if item in max_score:
                    if predictions[0][elem][0] > max_score[item]:
                        max_score[item] = predictions[0][elem][0]
                        my_ner[item] = batchner_length[elem]
                else:
                    max_score[item] = predictions[0][elem][0]
                    my_ner[item] = batchner_length[elem]
            print(my_ner)        
        f1 = open("src-test-features.txt","r").readlines()
        f2 = open("test_sentences.txt","r").readlines()
        modified_file = open("new_encoding.txt","w")
        for line in range(len(f1)):
            my_sent = f2[line]
            adding = [unichr(65512) + 'O']*len(f1[line].split(" "))
            if my_sent in my_ner:
                adding[my_ner[my_sent][0]] = unichr(65512) + 'B'
                for j in my_ner[my_sent][1:]:
                    adding[j] = unichr(65512) + 'I'
            splitted = f1[line].split(" ")
            for j in range(len(splitted)):
                modified_file.write(splitted[j].rstrip() + adding[j] + " ")   
            modified_file.write("\n")         
    