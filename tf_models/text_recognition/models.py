import tensorflow as tf
import os
from functools import partial
from tqdm import tqdm


class CRNN:
    """
    Building TF Graph & Session for Text Recognition, CRNN

    Order
        1. _attach_cnn()
        2. _attach_rnn()
        3. _attach_transcription_network()
        4. _attach_loss()
        5. _attach_decoder()
        6. _attach_metric()
        7. _attach_optimizer()

    """
    def __init__(self, num_classes=10, channel=1, height=28):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        self.num_classes = num_classes
        self.channels = channel
        self.height = height

        self._to_build = ['cnn',
                          'rnn',
                          'transcription',
                          "loss",
                          "decoder",
                          'metric',
                          'optimizer']
        self._built = []

        self._initialize_placeholders()

    def build_graph(self):
        """
        CRNN의 Tensorflow Graph를 구성함

        :return:
        self
        """
        return (self._attach_cnn()
                ._attach_rnn()
                ._attach_transcription()
                ._attach_loss()
                ._attach_decoder()
                ._attach_metric()
                ._attach_optimizer())

    def initialize_variables(self):
        """
        CRNN의 Variable 중 Uninitialized된 Variable만 초기화함

        :return:
        """
        with self.graph.as_default():
            global_vars = tf.global_variables()

            is_not_initialized = self.session.run(
                [tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in
                                    zip(global_vars, is_not_initialized) if not f]

            if len(not_initialized_vars):
                self.session.run(tf.variables_initializer(not_initialized_vars))

    def fit_generator(self, train_gen, valid_gen=None,
                      num_epoch=100, learning_rate=0.001,
                      summary_path=None):
        self.initialize_variables()

        with self.graph.as_default():
            if not '_merged' in dir(self):
                self._merged = tf.summary.merge_all()

            if summary_path:
                os.makedirs(summary_path,exist_ok=True)
                train_writer = tf.summary.FileWriter(os.path.join(summary_path,'train'),
                                                     self.graph)
                test_writer = tf.summary.FileWriter(os.path.join(summary_path, 'test'))
            else:
                train_writer = None
                test_writer = None

        if valid_gen:
            valid_images, valid_labels = valid_gen[0]

        # Gathering necessary Tensor and operation
        _x, _is_train, _targets, _lr = self.graph.get_collection('train_inputs')
        ler = self.graph.get_collection(tf.GraphKeys.METRIC_VARIABLES)[0]
        loss = self.graph.get_collection(tf.GraphKeys.LOSSES)[0]
        train_op = self.graph.get_collection(tf.GraphKeys.TRAIN_OP)[0]

        for i in range(num_epoch):
            train_ler = 0
            train_loss = 0
            for idx in tqdm(range(len(train_gen))):
                batch_images, batch_labels = train_gen[idx]
                _, train_ler_, train_loss_= \
                    self.session.run([train_op, ler, loss], feed_dict={
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
                train_summary = self.session.run(self._merged,
                                                 feed_dict={
                                                      _x: batch_images,
                                                      _targets: batch_labels})
                train_writer.add_summary(train_summary, i)
                if valid_gen:
                    test_summary = self.session.run(self._merged,
                                                    feed_dict={
                                                         _x: valid_images,
                                                         _targets: valid_labels})
                    test_writer.add_summary(test_summary, i)

            print("[{:3d}epoch]\nTrain ler :{:.3f} loss : {:.3f}"
                  .format(i, train_ler, train_loss))

            if valid_gen:
                val_ler, val_loss = self.session.run([ler, loss], feed_dict={
                    _x: valid_images,
                    _targets: valid_labels})
                print("Valid ler : {:.3f} loss : {:.3f}".format(val_ler, val_loss))
        return self

    def predict(self, images):
        return self.session.run(self._pred, feed_dict={self._x : images})

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

            with tf.variable_scope("dense-to-sparse"):
                # CTC Loss를 계산할 때는 Tensor가 아닌 SparseTensor를 이용해야해서
                # 형변환을 거쳐주는 작업
                indices = tf.where(tf.not_equal(self._targets, -1))
                values = tf.gather_nd(self._targets, indices)
                shape = tf.cast(tf.shape(self._targets), dtype=tf.int64)
                self._sparse_targets = tf.SparseTensor(indices, values, shape)

            tf.add_to_collection('train_inputs', self._x)
            tf.add_to_collection('train_inputs', self._is_train)
            tf.add_to_collection('train_inputs', self._targets)
            tf.add_to_collection('train_inputs', self._lr)

    def _attach_cnn(self, num_features=64):
        if 'cnn' in self._built:
            print("cnn network is already built")
            return self

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

        self._built.append(self._to_build.pop(0))
        return self

    def _attach_rnn(self, num_features=256, num_depth=2):
        if 'rnn' in self._built:
            print("rnn network is already built")
            return self
        elif not 'rnn' == self._to_build[0]:
            raise IndexError(
                "you should build {} network".format(
                    self._to_build[0]))

        with self.graph.as_default():
            with tf.variable_scope('recurrent_layers'):
                outputs = self._cnn_features
                for i in range(num_depth):
                    with tf.variable_scope('bidirectional_{}'.format(i+1)):
                        bidirectional = tf.keras.layers.Bidirectional
                        LSTM = tf.keras.layers.LSTM
                        outputs = bidirectional(
                            LSTM(num_features,
                                 return_sequences=True))(outputs)
            self._rnn_seqs = tf.identity(outputs, name='rnn_sequences')

        self._built.append(self._to_build.pop(0))
        return self

    def _attach_transcription(self):
        if 'transcription' in self._built:
            print("transcription network is already built")
            return self
        elif not 'transcription' == self._to_build[0]:
            raise IndexError(
                "you should build {} network".format(
                    self._to_build[0]))

        with self.graph.as_default():
            with tf.variable_scope('transcription'):
                with tf.variable_scope('sequence_length'):
                    shape = tf.shape(self._rnn_seqs)
                    batch_size, max_len, _ = tf.split(shape, 3, axis=0)
                    seq_len = tf.ones(batch_size, tf.int32) * max_len

                logits = tf.layers.Dense(self.num_classes + 1,
                                         name='logits')(self._rnn_seqs)

                self._logits_tp = tf.transpose(logits, [1, 0, 2])
                self._seq_len = seq_len

        self._built.append(self._to_build.pop(0))
        return self

    def _attach_loss(self):
        if 'loss' in self._built:
            print("loss network is already built")
            return self
        elif not 'loss' == self._to_build[0]:
            raise IndexError(
                "you should build {} network".format(
                    self._to_build[0]))

        with self.graph.as_default():
            # Calculate Loss by CTC
            with tf.variable_scope('losses'):
                ctc_loss = tf.nn.ctc_loss(
                    self._sparse_targets, self._logits_tp, self._seq_len)

            self._loss = tf.reduce_mean(ctc_loss, name='ctc_loss')
            tf.add_to_collection(tf.GraphKeys.LOSSES, self._loss)
            tf.summary.scalar('loss', self._loss)

        self._built.append(self._to_build.pop(0))
        return self

    def _attach_decoder(self):
        if 'decoder' in self._built:
            print("decoder network is already built")
            return self
        elif not 'decoder' == self._to_build[0]:
            raise IndexError(
                "you should build {} network".format(
                    self._to_build[0]))

        with self.graph.as_default():
            # Calculate the best path by Greedy Algorithm
            with tf.variable_scope('decoder'):
                decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(self._logits_tp,
                                                                   sequence_length=self._seq_len)
                pred = tf.sparse.to_dense(decoded[0])

            self._decoded = decoded[0]
            self._pred = tf.identity(pred, name='prediction')
            self._neg_logit = tf.identity(neg_sum_logits, name='prediction_score')

        self._built.append(self._to_build.pop(0))
        return self

    def _attach_metric(self):
        if 'metric' in self._built:
            print("metric network is already built")
            return self
        elif not 'metric' == self._to_build[0]:
            raise IndexError(
                "you should build {} network".format(
                    self._to_build[0]))

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

        self._built.append(self._to_build.pop(0))
        return self

    def _attach_optimizer(self, weight_decay=1e-5):
        if 'optimizer' in self._built:
            print("optimizer network is already built")
            return self
        elif not 'optimizer' == self._to_build[0]:
            raise IndexError(
                "you should build {} network".format(
                    self._to_build[0]))

        with self.graph.as_default():
            with tf.variable_scope('optimizer'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.variable_scope("l2_loss"):
                    weights = [var
                               for var in tf.trainable_variables()
                               if not "bias" in var.name]
                    l2_losses = tf.add_n([tf.nn.l2_loss(var) for var in weights], name='l2_losses')
                loss = self._loss + weight_decay * l2_losses

                with tf.control_dependencies(update_ops):
                    self._train_op = (tf.train
                                      .AdamOptimizer(learning_rate=self._lr)
                                      .minimize(loss))

        self._built.append(self._to_build.pop(0))
        return self
