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

    print(x_train.keys())
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

    feature = tf.feature_column.numeric_column('Sepallength', default_value=0,
                                               normalizer_fn=normalization(mean, std))
    print(feature)
    in_Sepallength = tf.feature_column.input_layer(features, [feature])
    print(in_Sepallength)

    with tf.Session() as sess:
        print(sess.run(in_Sepallength))

    data = pd.read_csv('/Users/songfeng/workspace/github/tensorflowDemo/data/categorical_column.csv',
                       names=['id', 'cat'])

    features_new = {key: np.array(value) for key, value in dict(data).items()}
    print(features_new)

    ci = tf.feature_column.categorical_column_with_identity('id', 10)
    ci_int = tf.feature_column.indicator_column(ci)
    # 为什么 要添加 这个步骤呢？因为 input_layer的入参是有要求的，
    # 如上所说必须是继承于DenseColumn的类型,详看 下面的 解释

    in_ci = tf.feature_column.input_layer(features_new, [ci_int])
    print(in_ci)

    with tf.Session() as sess:
        print(sess.run(in_ci))

    cate_vf = tf.feature_column.categorical_column_with_vocabulary_file('cat',
                                                                        vocabulary_file='/Users/songfeng/workspace/github/tensorflowDemo/data/vocabulary_file.csv',
                                                                        vocabulary_size=2,
                                                                        num_oov_buckets=3, )

    ind_cat = tf.feature_column.indicator_column(cate_vf)
    ind_cat

    inp_cat = tf.feature_column.input_layer(features_new, [ind_cat])
    inp_cat

    with tf.Session() as sess:
        # 在此处 必须使用 tf.tables_initializer来初始化 lookuptable
        sess.run(tf.tables_initializer())
        print(sess.run(inp_cat))

    cate_vl = tf.feature_column.categorical_column_with_vocabulary_list('cat',
                                                                        vocabulary_list=['test', 'train', 'eval'],
                                                                        default_value=1)
    print(cate_vl)
    ind_cat_l = tf.feature_column.indicator_column(cate_vl)

    inp_cat_l = tf.feature_column.input_layer(features_new, [ind_cat_l])

    with tf.Session() as sess:
        # 在此处 必须使用 tf.tables_initializer来初始化 lookuptable
        sess.run(tf.tables_initializer())
        print(sess.run(inp_cat_l))

    columns_name_multi = ['id', 'cat']
    data_multi = pd.read_csv('/Users/songfeng/workspace/github/tensorflowDemo/data/categorical_column_multi.csv',
                             names=columns_name_multi)
    # data_multi['cat'] = data_multi['cat'].map(lambda x: x.split(" "))
    print(data_multi)
    features_multi = {key: np.array(value) for key, value in dict(data_multi).items()}
    print(features_multi)
    cate_multi = tf.feature_column.categorical_column_with_vocabulary_list('cat',
                                                                                    vocabulary_list=['test', 'train',
                                                                                                     'eval'],
                                                                                    default_value=1)
    print(cate_multi)
    cate_multi = tf.feature_column.indicator_column(cate_multi)

    cate_multi = tf.feature_column.input_layer(features_multi, [cate_multi])

    with tf.Session() as sess:
        # 在此处 必须使用 tf.tables_initializer来初始化 lookuptable
        sess.run(tf.tables_initializer())
        print(sess.run(cate_multi))

    cate_hb = tf.feature_column.categorical_column_with_hash_bucket('id', hash_bucket_size=5, dtype=tf.int64)
    ind_cat_hb = tf.feature_column.indicator_column(cate_hb)
    inp_cat_hb = tf.feature_column.input_layer(features_new, [ind_cat_hb])
    with tf.Session() as sess:
        print(sess.run(inp_cat_hb))

    print(features_new)
