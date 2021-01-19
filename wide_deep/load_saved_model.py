import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
print(tf.__version__)

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

if __name__ == '__main__':

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

    with tf.Session() as sess:
        model = tf.saved_model.loader.load(sess=sess,
                                              tags=['serve'],
                                              export_dir="/Users/songfeng/workspace/github/tensorflowDemo/model/widedeep3_pb/1610354407")

        print("hello")
        print(model.fit(test))
        print("hello")




