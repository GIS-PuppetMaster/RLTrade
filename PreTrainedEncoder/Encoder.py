import tensorflow as tf
from tensorflow.layers import *
from tensorflow.nn import conv1d_transpose
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Util.Util import gelu
windows_size = 60
batch_size = 128

sess = tf.Session()
with tf.name_scope("encoder"):
    encoder_input = tf.placeholder(tf.float32, [None, 60, 6], name='encoder_input')
    x_1 = Conv1D(128, 3, dilation_rate=4, padding="same",activation=gelu)(encoder_input)
    x_2 = Conv1D(128, 3, dilation_rate=2, padding="same",activation=gelu)(x_1)
    x_3 = Conv1D(64, 5, dilation_rate=2, padding="valid",activation=gelu)(x_2)
    x_4 = Conv1D(64, 5, dilation_rate=4, padding="valid",activation=gelu)(x_3)
    x_5 = Conv1D(32, 7, dilation_rate=2, padding="valid",activation=gelu)(x_4)
    encoder_output = Conv1D(16, 7, dilation_rate=2, padding="valid",activation=gelu)(x_5)
with tf.name_scope("decoder"):
    decoder_input = conv1d_transpose(encoder_output, filters=tf.random.normal(shape=[7, 32, 16]),
                                     output_shape=tf.shape(x_5), dilations=2, padding="VALID", strides=1,activation=gelu)
    x_1 = conv1d_transpose(decoder_input, filters=tf.random.normal(shape=[5, 64, 32]), output_shape=tf.shape(x_4),
                           dilations=4, padding="VALID", strides=1,activation=gelu)
    x_2 = conv1d_transpose(x_1, filters=tf.random.normal(shape=[5, 64, 64]), output_shape=tf.shape(x_3), dilations=2,
                           padding="VALID", strides=1,activation=gelu)
    x_3 = conv1d_transpose(x_2, filters=tf.random.normal(shape=[3, 128, 64]), output_shape=tf.shape(x_2), dilations=2,
                           padding="SAME", strides=1,activation=gelu)
    x_4 = conv1d_transpose(x_3, filters=tf.random.normal(shape=[3, 128, 128]), output_shape=tf.shape(x_1), dilations=4,
                           padding="SAME", strides=1,activation=gelu)
    decoder_output = conv1d_transpose(x_4, filters=tf.random.normal(shape=[3, 6, 128]),
                                      output_shape=tf.shape(encoder_input), strides=1,activation=gelu)

file_list = [file_name for file_name in os.listdir('../Data/train/') if 'day' in file_name]
data = {}
for file_name in file_list:
    x = pd.read_csv('../Data/train/' + file_name).values[:, 1:]
    scaler = StandardScaler()
    x= scaler.fit_transform(x)
    data[file_name] = (x, scaler)
# train
for dataset, scaler in data.values():
    for i in range(dataset.shape[0] // batch_size):
        if (i + 1) * batch_size < dataset.shape[0]:
            batch = dataset[i * batch_size:(i + 1) * batch_size, ...]
        else:
            batch = dataset[i * batch_size:, ...]
