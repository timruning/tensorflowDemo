import tensorflow as tf
import pandas as pd
import numpy as np


def normalization(mean, std):
    def normalizer_fn(x):
        return (x - mean) / std

    return normalizer_fn


if __name__ == '__main__':
    columns_name = ['Sepallength', 'Sepalwidth', 'Petallength', 'Petalwidth', 'label']
    data = pd.read_csv('/Users/songfeng/workspace/github/tensorflowDemo/data/iris.csv')
    col_map = {"sepal.length": 'Sepallength', "sepal.width": 'Sepalwidth', "petal.length": 'Petallength',
               "petal.width": 'Petalwidth', "variety":
                   'label'}
    data = data.rename(index=str, columns=col_map)
    x_train, y_train = data.iloc[:, 0:4], data.iloc[:, -1:]
    print("x")
    print(x_train.head(5))
    print("y")
    print(y_train.head(5))
    features = {key: np.array(value) for key, value in dict(x_train).items()}
    target = {key: np.array(value) for key, value in dict(y_train).items()}

    x_train.keys()
    feature_columns = [tf.feature_column.numeric_column(key, default_value=0) for key in x_train.keys()]
    # 使用 input_layer 作为model的一个 input layer
    inn = tf.feature_column.input_layer(features, feature_columns)
    print(inn)

    tf.feature_column.input_layer(
        features,
        feature_columns,
        weight_collections=None,
        trainable=True,
        cols_to_vars=None,
        cols_to_output_tensors=None
    )

    with tf.Session() as sess:
        print(sess.run(inn))

    # 比如标准化

    mean = x_train['Sepallength'].mean()
    std = x_train['Sepallength'].std()
    print('mean is {},std is {}'.format(mean, std))

    feature = tf.feature_column.numeric_column('Sepallength', default_value=0, normalizer_fn=normalization(mean, std))
    print(feature)

    in_Sepallength = tf.feature_column.input_layer(features, [feature])
    print(in_Sepallength)

    with tf.Session() as sess:
        print(sess.run(in_Sepallength))
