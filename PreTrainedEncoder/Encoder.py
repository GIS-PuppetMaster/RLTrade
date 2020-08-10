import tensorflow as tf
from tensorflow.layers import *
from tensorflow.nn import conv1d_transpose
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Util.Util import gelu
import numpy as np
from tensorboardX import SummaryWriter

windows_size = 60
batch_size = 128
epochs = 100

sess = tf.Session()
with tf.variable_scope("encoder"):
    encoder_input = tf.placeholder(tf.float32, [None, 60, 6], name='encoder_input')
    x_1 = Conv1D(128, 3, dilation_rate=4, padding="same", activation=gelu)(encoder_input)
    x_2 = Conv1D(128, 3, dilation_rate=2, padding="same", activation=gelu)(x_1)
    x_3 = Conv1D(64, 5, dilation_rate=2, padding="valid", activation=gelu)(x_2)
    x_4 = Conv1D(64, 5, dilation_rate=4, padding="valid", activation=gelu)(x_3)
    x_5 = Conv1D(32, 7, dilation_rate=2, padding="valid", activation=gelu)(x_4)
    encoder_output = Conv1D(16, 7, dilation_rate=2, padding="valid", activation=gelu)(x_5)
with tf.variable_scope("decoder"):
    decoder_1 = gelu(conv1d_transpose(encoder_output, filters=tf.random.normal(shape=[7, 32, 16]),
                                      output_shape=tf.shape(x_5), dilations=2, padding="VALID", strides=1))
    decoder_2 = gelu(
        conv1d_transpose(decoder_1, filters=tf.random.normal(shape=[7, 64, 32]), output_shape=tf.shape(x_4),
                         dilations=2, padding="VALID", strides=1))
    decoder_3 = gelu(
        conv1d_transpose(decoder_2, filters=tf.random.normal(shape=[5, 64, 64]), output_shape=tf.shape(x_3),
                         dilations=4, padding="VALID", strides=1))
    decoder_4 = gelu(
        conv1d_transpose(decoder_3, filters=tf.random.normal(shape=[5, 128, 64]), output_shape=tf.shape(x_2),
                         dilations=2, padding="VALID", strides=1))
    decoder_5 = gelu(
        conv1d_transpose(decoder_4, filters=tf.random.normal(shape=[3, 128, 128]), output_shape=tf.shape(x_1),
                         dilations=2, padding="SAME", strides=1))
    decoder_output = gelu(conv1d_transpose(decoder_5, filters=tf.random.normal(shape=[3, 6, 128]),
                                           output_shape=tf.shape(encoder_input), dilations=4, padding='SAME',
                                           strides=1))
loss = tf.reduce_mean(tf.losses.huber_loss(labels=encoder_input, predictions=decoder_output))
opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
saver = tf.train.Saver(var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES, scope='encoder')])
sess.run(tf.global_variables_initializer())
sw = SummaryWriter(flush_secs=5)


# data prepare:
def gen_data(path):
    file_list = [file_name for file_name in os.listdir(path) if 'day' in file_name]
    data = {}
    for file_name in file_list:
        x = pd.read_csv(path + file_name).values[:, 1:]
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        data[file_name] = (x, scaler)
    final_data = []
    for dataset_name in data.keys():
        dataset, scaler = data[dataset_name]
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
    sw.add_scalar('loss_value', loss_value, global_step=epoch)
    print(f'epoch:{epoch}, loss_value:{loss_value}')
    # test
    test_loss = 0
    for batch in test_data:
        batch_loss = sess.run([loss, opt], feed_dict={encoder_input: batch})[0]
        test_loss += batch_loss
    test_loss /= test_batch_num
    sw.add_scalar('test_loss_value', test_loss, global_step=epoch)
    if test_loss < min_loss_value:
        min_loss_value = test_loss
        saver.save(sess, "/checkpoint/", global_step=epoch)
