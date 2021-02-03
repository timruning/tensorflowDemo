# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print(tf.__version__)

URL = '../data/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
print(dataframe.head())


def label(ahd):
    if 'Yes'.__eq__(ahd):
        return 1
    else:
        return 0


dataframe['target'] = dataframe['AHD'].map(lambda x: label(x))
dataframe = dataframe[
    ['target', "Age", "Sex", "ChestPain", "RestBP", "Chol", "Fbs", "RestECG", "MaxHR", "ExAng", "Oldpeak", "Slope",
     "Ca", "Thal"]]
dataframe = dataframe.dropna()

# @title 默认标题文本
train, test = train_test_split(dataframe, shuffle=False, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

print(dict(dataframe)['Age'])


# 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5  # 小批量大小用于演示
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['Age'])
    print('A batch of targets:', label_batch)
    break

# 我们将使用该批数据演示几种特征列
example_batch = next(iter(train_ds))[0]

sess = tf.compat.v1.Session()


# 用于创建一个特征列
# 并转换一批次数据的一个实用程序方法
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    # print(feature_layer(example_batch).eval(session=sess))


age = feature_column.numeric_column("Age")
# demo(age)

age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 120])
# demo(age_buckets)

thal = feature_column.categorical_column_with_vocabulary_list(
    'Thal', ['fixed', 'normal', 'reversible'])

thal_one_hot = feature_column.indicator_column(thal)
# demo(thal_one_hot)

# 注意到嵌入列的输入是我们之前创建的类别列
thal_embedding = feature_column.embedding_column(thal, dimension=8)
# demo(thal_embedding)

thal_hashed = feature_column.categorical_column_with_hash_bucket(
    'Thal', hash_bucket_size=1000)
# demo(feature_column.indicator_column(thal_hashed))

crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
# demo(feature_column.indicator_column(crossed_feature))

feature_columns = []

line_columns = []
dnn_columns = []

# 数值列
for header in ['Age', 'Sex', 'Chol', 'Fbs', 'Oldpeak', 'Slope', 'Ca']:
    col = feature_column.numeric_column(header)
    feature_columns.append(col)
    line_columns.append(col)
    dnn_columns.append(col)

# 分桶列
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)
line_columns.append(age_buckets)
dnn_columns.append(age_buckets)

# 分类列
thal = feature_column.categorical_column_with_vocabulary_list(
    'Thal', ['fixed', 'normal', 'reversible'], dtype=tf.string)
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)
line_columns.append(thal_one_hot)
dnn_columns.append(thal_one_hot)

# 嵌入列
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)
line_columns.append(thal_embedding)
dnn_columns.append(thal_embedding)

# 组合列
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)
line_columns.append(crossed_feature)

demo(crossed_feature)

# feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# print(feature_layer(example_batch).numpy())
# print(feature_layer(example_batch).numpy().shape)
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

adagrad = tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1)

ftrl = tf.keras.optimizers.Ftrl(learning_rate=0.01)

DNNLinearCombinedRegressor = tf.estimator.DNNLinearCombinedRegressor

# estimator = DNNLinearCombinedRegressor(
#     # wide settings
#     linear_feature_columns=line_columns,
#     linear_optimizer=tf.train.Ftrl(learning_rate=0.01),
#     # deep settings
#     dnn_feature_columns=dnn_columns,
#     dnn_hidden_units=[1000, 500, 100],
#     dnn_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     # warm-start settings
#     model_dir="/Users/songfeng/workspace/github/tensorflowDemo/model/widedeep3",
#     # warm_start_from="/Users/songfeng/workspace/github/tensorflowDemo/model/widedeep2"
# )

estimator = DNNLinearCombinedRegressor(
    # wide settings
    linear_feature_columns=line_columns,
    linear_optimizer='Ftrl',
    # deep settings
    dnn_feature_columns=dnn_columns,
    dnn_hidden_units=[1000, 500, 100],
    dnn_optimizer='Adam',
    # warm-start settings
    # warm_start_from="/Users/songfeng/workspace/github/tensorflowDemo/model"
    model_dir="/Users/songfeng/workspace/github/tensorflowDemo/model/widedeep3"
    # warm_start_from="/Users/songfeng/workspace/github/tensorflowDemo/model/widedeep4"
)

# To apply L1 and L2 regularization, you can set dnn_optimizer to:

# To apply learning rate decay, you can set dnn_optimizer to a callable:
lambda: tf.AdamOptimizer(
    learning_rate=tf.exponential_decay(
        learning_rate=0.1,
        global_step=tf.get_global_step(),
        decay_steps=10000,
        decay_rate=0.96))

dnn_optimizer = lambda: tf.keras.optimizers.Adam(
    learning_rate=tf.compat.v1.train.exponential_decay(
        learning_rate=0.1,
        global_step=tf.compat.v1.train.get_global_step(),
        decay_steps=10000,
        decay_rate=0.96))

estimator.train(input_fn=lambda: df_to_dataset(train, batch_size=batch_size), steps=1000)
metrics = estimator.evaluate(input_fn=lambda: df_to_dataset(test, batch_size=batch_size))
print(metrics)

print("predict")
s = estimator.predict(input_fn=lambda: df_to_dataset(test, batch_size=batch_size))
for i in s:
    print(i)
    break

# estimator.export_saved_model()
['Age', 'Sex', 'Chol', 'Fbs', 'Oldpeak', 'Slope', 'Ca']


def serving_input_fn():
    label_ids = tf.compat.v1.placeholder(tf.int32, [None], name='target')
    Age_ids = tf.compat.v1.placeholder(tf.int32, [None], name='Age')
    Sex_ids = tf.compat.v1.placeholder(tf.int32, [None], name='Sex')
    Chol_ids = tf.compat.v1.placeholder(tf.int32, [None], name='Chol')
    Fbs_ids = tf.compat.v1.placeholder(tf.int32, [None], name='Fbs')
    Oldpeak_ids = tf.compat.v1.placeholder(tf.int32, [None], name='Oldpeak')
    Slope_ids = tf.compat.v1.placeholder(tf.int32, [None], name='Slope')
    Ca_ids = tf.compat.v1.placeholder(tf.int32, [None], name='Ca')
    Thal_ids = tf.compat.v1.placeholder(tf.string, [None], name='Thal')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        # 'target': label_ids,
        'Age': Age_ids,
        'Sex': Sex_ids,
        'Chol': Chol_ids,
        'Fbs': Fbs_ids,
        'Oldpeak': Oldpeak_ids,
        'Slope': Slope_ids,
        'Ca': Ca_ids,
        'Thal': Thal_ids,
    })()
    return input_fn


estimator.export_saved_model(export_dir_base="/Users/songfeng/workspace/github/tensorflowDemo/model/widedeep3_pb",
                             serving_input_receiver_fn=serving_input_fn)
