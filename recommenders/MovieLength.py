import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

# Ratings data.
ratings = tfds.load("movie_lens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movie_lens/100k-movies", split="train")

for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)

for x in movies.take(1).as_numpy_iterator():
    pprint.pprint(x)

ratings = ratings.map(lambda x: {
    "movie_id": x["movie_id"],
    "user_id": x["user_id"],
})
movies = movies.map(lambda x: {
    "movie_id": x["movie_id"],
    "movie_title": x["movie_title"],
})

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_ids = movies.batch(1_000).map(lambda x: x["movie_id"])
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_ids = np.unique(np.concatenate(list(movie_ids)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

# We convert bytes to strings since bytes are not serializable
unique_movie_id_strings = [id.decode("utf-8") for id in unique_movie_ids]
unique_user_id_strings = [id.decode("utf-8") for id in unique_user_ids]

unique_movie_id_strings[:10]

embedding_dimension = 32


class UserModel(tf.keras.Model):

    def __init__(self, embedding_dimension):
        super(UserModel, self).__init__()
        # The model itself is a single embedding layer.
        # However, we could expand this to an arbitrarily complicated Keras model, as long
        # as the output is an vector `embedding_dimension` wide.
        user_features = [tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                "user_id", unique_user_id_strings,
            ),
            embedding_dimension,
        )]
        self.embedding_layer = tf.keras.layers.DenseFeatures(user_features, name="user_embedding")

    def call(self, inputs):
        return self.embedding_layer(inputs)


# We initialize these models and later pass them to the full model.
user_model = UserModel(embedding_dimension)


class MovieModel(tf.keras.Model):

    def __init__(self, embedding_dimension):
        super(MovieModel, self).__init__()
        movie_features = [tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                "movie_id", unique_movie_id_strings,
            ),
            embedding_dimension,
        )]
        self.embedding_layer = tf.keras.layers.DenseFeatures(movie_features, name="movie_embedding")

    def call(self, inputs):
        return self.embedding_layer(inputs)


movie_model = MovieModel(embedding_dimension)

metrics = tfrs.metrics.FactorizedTopK(
    candidates=movies.batch(128).map(lambda x: {"movie_id": x["movie_id"]}).map(movie_model)
)

task = tfrs.tasks.RetrievalTask(
    corpus_metrics=metrics
)
