import tensorflow as tf
from tensorflow.layers import *
from tensorflow.nn import conv1d_transpose
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Util.Util import *
import numpy as np
from tensorboardX import SummaryWriter
import argparse
from tensorflow.contrib.layers import xavier_initializer

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=bool, default=False)
args = parser.parse_args()

windows_size = 60
batch_size = 128
epochs = 100

sess = tf.Session()
with tf.variable_scope("encoder"):
    encoder_input = tf.placeholder(tf.float32, [None, 60, 6], name='encoder_input')
    x_1 = Conv1D(128, 3, dilation_rate=4, padding="same", activation=gelu)(encoder_input)
    x_1_ = MaxPooling1D(2, 1)(x_1)
    x_2 = Conv1D(128, 3, dilation_rate=2, padding="same", activation=gelu)(x_1_)
    x_2_ = MaxPooling1D(2, 1)(x_2)
    x_3 = Conv1D(64, 5, dilation_rate=2, padding="valid", activation=gelu)(x_2_)
    x_3_ = MaxPooling1D(2, 1)(x_3)
    x_4 = Conv1D(64, 5, dilation_rate=4, padding="valid", activation=gelu)(x_3_)
    x_4_ = MaxPooling1D(2, 1)(x_4)
    x_5 = Conv1D(32, 7, dilation_rate=2, padding="valid", activation=gelu)(x_4_)
    x_5_ = MaxPooling1D(2, 1)(x_5)
    x_6 = Conv1D(16, 7, dilation_rate=2, padding="valid", activation=gelu)(x_5_)
    encoder_output = MaxPooling1D(2, 1)(x_6)
with tf.variable_scope("decoder"):
    decoder_0_ = gelu(
        conv1d_transpose(encoder_output,
                         filters=tf.get_variable(name='decoder_0', shape=[1, 16, 16], initializer=xavier_initializer()),
                         output_shape=tf.shape(x_6), dilations=2, padding="VALID", strides=1))
    decoder_1 = gelu(conv1d_transpose(decoder_0_, filters=tf.get_variable(name='decoder_1', shape=[7, 32, 16],
                                                                          initializer=xavier_initializer()),
                                      output_shape=tf.shape(x_5_), dilations=2, padding="VALID", strides=1))
    decoder_1_ = gelu(
        conv1d_transpose(decoder_1,
                         filters=tf.get_variable(name='decoder_1_', shape=[1, 32, 32],
                                                 initializer=xavier_initializer()),
                         output_shape=tf.shape(x_5), dilations=2, padding="VALID", strides=1))
    decoder_2 = gelu(
        conv1d_transpose(decoder_1_,
                         filters=tf.get_variable(name='decoder_2', shape=[7, 64, 32], initializer=xavier_initializer()),
                         output_shape=tf.shape(x_4_),
                         dilations=2, padding="VALID", strides=1))
    decoder_2_ = gelu(
        conv1d_transpose(decoder_2,
                         filters=tf.get_variable(name='decoder_2_', shape=[1, 64, 64],
                                                 initializer=xavier_initializer()),
                         output_shape=tf.shape(x_4), dilations=2, padding="VALID", strides=1))
    decoder_3 = gelu(
        conv1d_transpose(decoder_2_,
                         filters=tf.get_variable(name='decoder_3', shape=[5, 64, 64], initializer=xavier_initializer()),
                         output_shape=tf.shape(x_3_), dilations=4, padding="VALID", strides=1))
    decoder_3_ = gelu(
        conv1d_transpose(decoder_3, filters=tf.get_variable(name='decoder_3_', shape=[1, 64, 64],
                                                            initializer=xavier_initializer()),
                         output_shape=tf.shape(x_3), dilations=2, padding="VALID", strides=1))

    decoder_4 = gelu(
        conv1d_transpose(decoder_3_, filters=tf.get_variable(name='decoder_4', shape=[5, 128, 64],
                                                             initializer=xavier_initializer()),
                         output_shape=tf.shape(x_2_), dilations=2, padding="VALID", strides=1))
    decoder_4_ = gelu(
        conv1d_transpose(decoder_4, filters=tf.get_variable(name='decoder_4_', shape=[1, 128, 128],
                                                            initializer=xavier_initializer()),
                         output_shape=tf.shape(x_2), dilations=2, padding="VALID", strides=1))
    decoder_5 = gelu(
        conv1d_transpose(decoder_4_, filters=tf.get_variable(name='decoder_5', shape=[3, 128, 128],
                                                             initializer=xavier_initializer()),
                         output_shape=tf.shape(x_1_), dilations=2, padding="SAME", strides=1))
    decoder_5_ = gelu(
        conv1d_transpose(decoder_5, filters=tf.get_variable(name='decoder_5_', shape=[1, 128, 128],
                                                            initializer=xavier_initializer()),
                         output_shape=tf.shape(x_1), dilations=2, padding="VALID", strides=1))

    decoder_output = gelu(conv1d_transpose(decoder_5_, filters=tf.get_variable(name='decoder_output', shape=[3, 6, 128],
                                                                               initializer=xavier_initializer()),
                                           output_shape=tf.shape(encoder_input), dilations=4, padding='SAME', strides=1))
loss = tf.reduce_mean(tf.losses.huber_loss(labels=encoder_input, predictions=decoder_output))
opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
encoder_saver = tf.train.Saver(var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES, scope='encoder')])
decoder_saver = tf.train.Saver(var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES, scope='decoder')])
sess.run(tf.global_variables_initializer())
if args.test:
    encoder_saver.restore(sess, "./checkpoints/encoder/")
    decoder_saver.restore(sess, "./checkpoints/decoder/")
sw = SummaryWriter(flush_secs=5)


# data prepare:
def gen_data(path):
    file_list = [file_name for file_name in os.listdir(path) if 'day' in file_name]
    data = {}
    # scaler = StandardScaler()
    for file_name in file_list:
        x = pd.read_csv(path + file_name).values[:, 1:]
        data[file_name] = x
    dataset = np.concatenate(list(data.values()), axis=0)
    # dataset = scaler.fit_transform(dataset)
    dataset = log10plus1R(dataset)
    final_data = []
    for i in range(0, dataset.shape[0], batch_size):
        # i 为当前batch序列起点
        batch = []
        # 生成batch
        for j in range(i, i + batch_size):
            # j为当前sample序列起点
            if j + 60 <= dataset.shape[0]:
                batch.append(np.expand_dims(dataset[j:j + 60, ...], axis=0))
            else:
                tmp = dataset[j:, ...]
                batch.append(
                    np.expand_dims(np.pad(tmp, ((windows_size - tmp.shape[0], 0), (0, 0)), constant_values=(0,)),
                                   axis=0))
        final_data.append(np.concatenate(batch, axis=0))
    np.random.shuffle(final_data)
    return final_data


train_data = gen_data('../Data/train/')
test_data = gen_data('../Data/test/')
train_batch_num = len(train_data)
test_batch_num = len(test_data)
# train
global_step = 0
if not os.path.exists('./checkpoints/encoder'):
    os.makedirs('./checkpoints/encoder')
if not os.path.exists('./checkpoints/decoder'):
    os.makedirs('./checkpoints/decoder')
for epoch in range(epochs):
    loss_value = 0
    sw.add_scalar('epoch', epoch, global_step=global_step)
    # 遍历batch
    min_loss_value = float('inf')
    for batch in train_data:
        batch_loss = sess.run([loss, opt], feed_dict={encoder_input: batch})[0]
        sw.add_scalar('train_batch_loss', batch_loss, global_step=global_step)
        loss_value += batch_loss
        global_step += 1
    loss_value /= train_batch_num
    sw.add_scalar('train_loss_value', loss_value, global_step=epoch)
    print(f'epoch:{epoch}, train_loss_value:{loss_value}')
    # test
    test_loss = 0
    for batch in test_data:
        batch_loss = sess.run([loss, opt], feed_dict={encoder_input: batch})[0]
        test_loss += batch_loss
    test_loss /= test_batch_num
    sw.add_scalar('test_loss_value', test_loss, global_step=epoch)
    if test_loss < min_loss_value:
        min_loss_value = test_loss
        encoder_saver.save(sess, "./checkpoints/encoder/", global_step=epoch)
        decoder_saver.save(sess, "./checkpoints/decoder/", global_step=epoch)
