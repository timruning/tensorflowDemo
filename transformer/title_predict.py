import os
import re
import tensorflow as tf
from tensorflow.python.platform import gfile
import pandas as pd
from datetime import datetime
from threading import Timer
import time


def process(line, word2id, max_sequence_length=15):
    line = line.lower()
    res = [0] * max_sequence_length

    words = line.split(' ')[:max_sequence_length]
    for i, word in enumerate(words):
        if word in word2id:
            res[i] = word2id[word]
        else:
            res[i] = 0

    return res


def read_word2id(file_name):
    word2id = {}
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub('\n', '', line)
            if len(line.split(' ')) == 2:
                word, id = line.split(' ')
                word2id[word] = int(id)
    return word2id


if __name__ == '__main__':
    # print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    model_path = "/Users/songfeng/workspace/github/tensorflowDemo/transformer/data/model_128.pb"
    word2id = read_word2id('/Users/songfeng/workspace/github/tensorflowDemo/transformer/data/word2id.txt')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    input_x = sess.graph.get_tensor_by_name('input_x:0')

    encoded_outputs = sess.graph.get_tensor_by_name('transformer-encoding/encoded_outputs:0')

    tf.compat.v1.summary.histogram('valid_embeddings', encoded_outputs)
    merged = tf.compat.v1.summary.merge_all()

    pred_data_path = "/Users/songfeng/workspace/github/tensorflowDemo/transformer/data/itemDF/part-00000-4e415f48" \
                     "-17bb-4515-a867-63827999bb87-c000.csv"
    pred_data = pd.read_csv(pred_data_path, sep="|")
    print(pred_data.shape)
    print(pred_data.head())

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    train_writer = tf.compat.v1.summary.FileWriter('/Users/songfeng/workspace/github/tensorflowDemo/transformer/log',
                                                   sess.graph)
    vec_list = []
    for i in range(pred_data.shape[0]):
        line = pred_data['pname'][i]
        x = process(str(line), word2id, max_sequence_length=15)
        _merge, ret = sess.run([merged, encoded_outputs], feed_dict={input_x: [x]}, options=run_options,
                               run_metadata=run_metadata)
        train_writer.add_summary(_merge, i)
        train_writer.add_run_metadata(run_metadata, 'step%d' % i)
        vec_list.append(','.join([str(k) for k in ret[0]]))
        if i % 10000 == 0:
            print(i)
            print(','.join([str(k) for k in ret[0]]))
    pred_data['vector'] = vec_list
    print(pred_data['vector'][0])
    print(len(pred_data['vector'][0]))
    print(pred_data.head)

    # print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
