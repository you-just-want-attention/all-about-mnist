import tensorflow as tf
import os
from functools import partial
from tqdm import tqdm


class CRNN:
    def __init__(self, num_classes=10, channel=1, height=28):
        self.num_classes = num_classes
        self.channels = channel
        self.height = height
        self.graph = tf.Graph()
        self._session = None

        self._initialize_placeholders()

    def build_graph(self):
        """
        CRNN의 Tensorflow Graph를 구성함

        :return:
        self
        """
        return (self._attach_cnn_network()
                ._attach_rnn_network()
                ._attach_transcription()
                ._attach_metric()
                ._attach_optimizer())

    def fit_generator(self, train_gen, valid_gen=None,
                      num_epoch=100, learning_rate=0.001,
                      summary_path=None):
        with self.graph.as_default():
            if self._session is None:
                self._session = self._initialize_session()

            if valid_gen:
                valid_images, valid_labels = valid_gen[0]

            if summary_path:
                train_writer = tf.summary.FileWriter(os.path.join(summary_path,'train'),
                                                     self.graph)
                test_writer = tf.summary.FileWriter(os.path.join(summary_path, 'test'))
            else:
                train_writer = None
                test_writer = None

            # Gathering necessary Tensor and operation
            _x, _is_train, _targets, _lr = tf.get_collection('train_inputs')
            ler = self.graph.get_collection(tf.GraphKeys.METRIC_VARIABLES)[0]
            loss = self.graph.get_collection(tf.GraphKeys.LOSSES)[0]
            train_op = self.graph.get_collection(tf.GraphKeys.TRAIN_OP)[0]

            for i in range(num_epoch):
                train_ler = 0
                train_loss = 0
                for idx in tqdm(range(len(train_gen))):
                    batch_images, batch_labels = train_gen[idx]
                    _, train_ler_, train_loss_= \
                        self._session.run([train_op, ler, loss], feed_dict={
                            _x : batch_images,
                            _targets : batch_labels,
                            _lr : learning_rate,
                            _is_train : True})
                    train_ler += train_ler_
                    train_loss += train_loss_
                train_gen.on_epoch_end()
                train_ler /= len(train_gen)
                train_loss /= len(train_gen)

                if train_writer:
                    train_summary = self._session.run(self._merged,
                                                      feed_dict={
                                                          _x: batch_images,
                                                          _targets: batch_labels})
                    train_writer.add_summary(train_summary, i)
                    if valid_gen:
                        test_summary = self._session.run(self._merged,
                                                         feed_dict={
                                                             _x: valid_images,
                                                             _targets: valid_labels})
                        test_writer.add_summary(test_summary, i)

                print("[{:3d}epoch]\nTrain ler :{:.3f} loss : {:.3f}"
                      .format(i, train_ler, train_loss))

                if valid_gen:
                    val_ler, val_loss = self._session.run([ler, loss], feed_dict={
                        _x: valid_images,
                        _targets: valid_labels})
                    print("Valid ler :{:.3f} loss : {:.3f}".format(i, val_ler, val_loss))
        return self._session

    def predict(self, images):
        return self._session.run(
            self._pred, feed_dict={
                self._x : images
            })

    def _initialize_placeholders(self):
        with self.graph.as_default():
            self._x = tf.placeholder(
                tf.float32,
                shape=(None, self.height, None, self.channels),
                name='image')
            self._is_train = tf.placeholder_with_default(False, None,
                                                         name='is_train')
            self._targets = tf.placeholder(tf.int32, shape=(None, None),
                                           name='targets')
            self._lr = tf.placeholder_with_default(0.001, None,
                                                   name='learning_rate')

            with tf.variable_scope("sparse-to-dense"):
                indices = tf.where(tf.not_equal(self._targets, -1))
                values = tf.gather_nd(self._targets, indices)
                shape = tf.cast(tf.shape(self._targets), dtype=tf.int64)
                self._sparse_targets = tf.SparseTensor(indices, values, shape)

            tf.add_to_collection('train_inputs', self._x)
            tf.add_to_collection('train_inputs', self._is_train)
            tf.add_to_collection('train_inputs', self._targets)
            tf.add_to_collection('train_inputs', self._lr)

    def _initialize_session(self):
        with self.graph.as_default():
            self._merged = tf.summary.merge_all()
            self._session = tf.Session()
            self._session.run(tf.global_variables_initializer())
        return self._session

    def _attach_cnn_network(self, num_features=64):
        with self.graph.as_default():
            with tf.variable_scope('convolution_layers'):
                conv_layer = partial(tf.layers.Conv2D,
                                     kernel_size=(3, 3),
                                     padding='same',
                                     activation=tf.nn.relu)
                maxpool_layer = partial(tf.layers.MaxPooling2D,
                                        pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='same')
                bn_layer = tf.layers.BatchNormalization

                transposed_x = tf.transpose(self._x, [0,2,1,3])
                conv1 = conv_layer(num_features, name='conv1')(transposed_x)
                max1 = maxpool_layer(name='maxpool1')(conv1)
                conv2 = conv_layer(num_features * 2, name='conv2')(max1)
                max2 = maxpool_layer(name='maxpool2')(conv2)
                conv3 = conv_layer(num_features * 4, name='conv3')(max2)
                conv4 = conv_layer(num_features * 4, name='conv4')(conv3)
                max3 = maxpool_layer(name='maxpool3')(conv4)

                conv5 = conv_layer(num_features * 8, name='conv5')(max3)
                bn1 = bn_layer(name='bn1')(conv5, training=self._is_train)
                conv6 = conv_layer(num_features * 8, name='conv6')(bn1)
                bn2 = bn_layer(name='bn2')(conv6, training=self._is_train)
                max4 = maxpool_layer(pool_size=(1, 2),
                                     strides=(1, 2),
                                     name='maxpool4')(bn2)

                conv_features = conv_layer(num_features * 8,
                                           kernel_size=(2, 2),
                                           name='conv7')(max4)

            with tf.variable_scope('map-to-sequence'):
                shape = tf.shape(conv_features)
                outputs = tf.reshape(
                    conv_features, shape=[
                        shape[0], shape[1], num_features * 16])
            self._cnn_features = tf.identity(outputs, name='cnn_features')

        return self

    def _attach_rnn_network(self, num_features=256, num_depth=2):
        with self.graph.as_default():
            with tf.variable_scope('recurrent_layers'):
                outputs = self._cnn_features
                for i in range(num_depth):
                    with tf.variable_scope('bidirectional_{}'.format(i)):
                        bidirectional = tf.keras.layers.Bidirectional
                        LSTM = tf.keras.layers.LSTM
                        outputs = bidirectional(
                            LSTM(num_features,
                                 return_sequences=True))(outputs)
                self._rnn_seqs = tf.identity(outputs, name='rnn_sequences')
        return self

    def _attach_transcription(self):
        with self.graph.as_default():
            with tf.variable_scope('transcription'):
                shape = tf.shape(self._cnn_features)
                batch_size, max_len, _ = tf.split(shape, 3, axis=0)
                seq_len = tf.ones(batch_size, tf.int32) * max_len

                # Calculate Loss by CTC
                logits = tf.layers.Dense(
                    self.num_classes + 1,
                    name='logits')(
                    self._rnn_seqs)
                logits_tp = tf.transpose(logits, [1, 0, 2])
                ctc_loss = tf.nn.ctc_loss(
                    self._sparse_targets, logits_tp, seq_len)

                # Calculate the best path by Greedy Algorithm
                decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(logits_tp,
                                                                   sequence_length=seq_len)
                pred = tf.sparse.to_dense(decoded[0])

            self._decoded = decoded[0]
            self._pred = tf.identity(pred, name='prediction')
            self._neg_logit = tf.identity(neg_sum_logits, name='prediction_score')

            self._loss = tf.reduce_mean(ctc_loss, name='ctc_loss')
            tf.add_to_collection(tf.GraphKeys.LOSSES, self._loss)
            tf.summary.scalar('loss', self._loss)

        return self

    def _attach_metric(self):
        with self.graph.as_default():
            with tf.variable_scope('metric'):
                label_error_rate = tf.reduce_mean(
                    tf.edit_distance(tf.cast(self._decoded, tf.int32),
                                     self._sparse_targets),
                    name='label_error_rate')
                tf.add_to_collection(
                    tf.GraphKeys.METRIC_VARIABLES,
                    label_error_rate)
                tf.summary.scalar('label_error_rate', label_error_rate)

        return self

    def _attach_optimizer(self):
        with self.graph.as_default():
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self._train_op = (tf.train
                                  .AdamOptimizer(learning_rate=self._lr)
                                  .minimize(self._loss))
        return self
