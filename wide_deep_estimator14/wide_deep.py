import tensorflow as tf

if __name__ == '__main__':

    numeric_feature = tf.feature_column.numeric_column(...)
    categorical_column_a = tf.feature_column.categorical_column_with_hash_bucket(...)
    categorical_column_b = tf.feature_column.categorical_column_with_hash_bucket(...)

    categorical_feature_a_x_categorical_feature_b = tf.feature_column.crossed_column(...)
    categorical_feature_a_emb = tf.feature_column.embedding_column(
        categorical_column=tf.feature_column.categorical_feature_a, ...)
    categorical_feature_b_emb = tf.feature_column.embedding_column(
        categorical_column=tf.feature_column.categorical_feature_b, ...)

    DNNLinearCombinedRegressor = tf.estimator.DNNLinearCombinedRegressor
    estimator = DNNLinearCombinedRegressor(
        # wide settings
        linear_feature_columns=[categorical_feature_a_x_categorical_feature_b],
        linear_optimizer=tf.train.FtrlOptimizer(...),
        # deep settings
        dnn_feature_columns=[
            categorical_feature_a_emb, categorical_feature_b_emb,
            numeric_feature],
        dnn_hidden_units=[1000, 500, 100],
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(...),
        # warm-start settings
        warm_start_from="/path/to/checkpoint/dir")

    # To apply L1 and L2 regularization, you can set dnn_optimizer to:
    tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001,
        l2_regularization_strength=0.001)
    # To apply learning rate decay, you can set dnn_optimizer to a callable:
    lambda: tf.AdamOptimizer(
        learning_rate=tf.exponential_decay(
            learning_rate=0.1,
            global_step=tf.get_global_step(),
            decay_steps=10000,
            decay_rate=0.96))


    # It is the same for linear_optimizer.

    # Input builders

    def input_fn_train:
        # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
        # index.
        pass


    def input_fn_eval:
        # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
        # index.
        pass


    def input_fn_predict:
        # Returns tf.data.Dataset of (x, None) tuple.
        pass


    estimator.train(input_fn=input_fn_train, steps=100)
    metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
    predictions = estimator.predict(input_fn=input_fn_predict)
