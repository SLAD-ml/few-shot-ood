from __future__ import print_function
import tensorflow as tf


class DistanceNetwork:
    def __init__(self):
        pass

    def __call__(self,
                 support_set,
                 input_sent,
                 name,
                 dropout_keep_prob=1.0,
                 is_training=None):

        target_shape_order = len(input_sent.get_shape().as_list())
        input_sent_ex = input_sent
        if target_shape_order == 2:
            input_sent_ex = tf.expand_dims(input_sent, axis=1)

        normalize_support_set = tf.nn.l2_normalize(support_set, axis=2)
        normalize_input_sent_ex = tf.nn.l2_normalize(input_sent_ex, axis=2)

        cosine_similarity = tf.matmul(normalize_input_sent_ex,
                                      tf.transpose(normalize_support_set,
                                                   [0, 2, 1]))
        if target_shape_order == 2:
            cosine_similarity = cosine_similarity[:, 0, :]
        return cosine_similarity


class Classify:
    def __init__(self):
        pass

    def __call__(self, similarities, name, softmax_factor=None):
        with tf.name_scope('classification' + name):
            similarities = softmax_factor * similarities
            softmax_similarities = tf.nn.softmax(similarities, axis=-1)
        return softmax_similarities


class CNNEncoder:
    def __init__(self, params):
        self.params = params

    def __call__(self, sent_input, support_set, reuse, is_training, mask):

        sent_input = tf.expand_dims(sent_input, axis=-1)
        tmp = tf.shape(sent_input)
        sent_input = tf.reshape(sent_input, [-1, self.params['max_length'], self.params['emb_size'], 1])
        sent_input = tf.transpose(sent_input, [0, 1, 3, 2])
        mask = tf.reshape(mask, [-1, tmp[-3]])

        with tf.variable_scope('g', reuse=reuse):
            g_conv1_encoder = tf.layers.conv2d(sent_input,
                                               self.params['hidden_size'],
                                               [self.params['filter_size'], 1],
                                               strides=(1, 1),
                                               padding='SAME',
                                               name='conv1',
                                               use_bias=False)
            g_conv1_encoder = tf.tanh(g_conv1_encoder)
            if self.params['enable_batchnorm'] is True:
                print('batch norm enabled.')
                g_conv1_encoder = tf.contrib.layers.batch_norm(g_conv1_encoder,
                                                               updates_collections=None,
                                                               decay=0.99,
                                                               scale=True,
                                                               center=True,
                                                               is_training=is_training)

            g_conv2_encoder = tf.squeeze(g_conv1_encoder, [2])

            lengths = tf.expand_dims(tf.reduce_sum(mask, axis=1), axis=1)
            mask = tf.expand_dims(mask, axis=2)
            g_conv2_encoder = g_conv2_encoder * tf.cast(mask, 'float32')
            g_conv_encoder = tf.reduce_sum(g_conv2_encoder, axis=1)
            g_conv_encoder = g_conv_encoder / tf.cast(lengths, 'float32')

            print('g_conv_encoder', g_conv_encoder)
            return g_conv_encoder


class biLSTMEncoder:
    def __init__(self, params):
        self.params = params

    def __call__(self, sent_input, support_set, reuse, is_training, mask):

        sent_input = tf.expand_dims(sent_input, axis=-1)
        tmp = tf.shape(sent_input)
        sent_input = tf.reshape(sent_input, 
                                 [-1, self.params['max_length'], self.params['emb_size']])

        mask = tf.reshape(mask, [-1, tmp[-3]])

        with tf.variable_scope('g', reuse=reuse):
            cell_fw = tf.contrib.rnn.LSTMCell(self.params['hidden_size']/2, name='fw')
            cell_bw = tf.contrib.rnn.LSTMCell(self.params['hidden_size']/2, name='bw')
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                         cell_bw, sent_input,
                                                         dtype='float32')
            g_conv_encoder = tf.concat([outputs[0], outputs[1]], axis=-1)

            if self.params['enable_batchnorm'] is True:
                print('batch norm enabled.')
                g_conv_encoder = tf.contrib.layers.batch_norm(g_conv_encoder,
                                                              updates_collections=None,
                                                              decay=0.99,
                                                              scale=True,
                                                              center=True,
                                                              is_training=is_training)

            lengths = tf.expand_dims(tf.reduce_sum(mask, axis=1), axis=1)
            mask = tf.expand_dims(mask, axis=2)
            g_conv_encoder = g_conv_encoder * tf.cast(mask, 'float32')
            g_conv_encoder = tf.reduce_sum(g_conv_encoder, axis=1)
            g_conv_encoder = g_conv_encoder / tf.cast(lengths, 'float32')

            print('g_conv_encoder', g_conv_encoder)
            return g_conv_encoder


class MatchingNetwork:
    def __init__(self, params):

        self.params = params
        self.dn = DistanceNetwork()
        self.classify = Classify()
        self.ss_g = CNNEncoder(params)
        # self.ss_g = biLSTMEncoder(params)
        self.W = tf.get_variable("W",
                                 dtype=tf.float32,
                                 initializer=tf.constant(self.params['wordvectors'].astype('float32')))

        sen_len = self.params['max_length']
        self.input_support_set_sents = tf.placeholder(tf.int32,
                                                       [None, None, sen_len],
                                                       'support_set_sents')
        self.support_set_sents = tf.nn.embedding_lookup(self.W,
                                                         self.input_support_set_sents)
        self.support_set_labels = tf.placeholder(tf.float32,
                                                 [None, None, None],
                                                 'support_set_labels')
        self.support_set_sents_mask = tf.placeholder(tf.float32,
                                                      [None, None, sen_len],
                                                      'support_set_sents_mask')

        self.input_target_sent = tf.placeholder(tf.int32,
                                                 [None, sen_len],
                                                 'target_sent')
        self.target_sent = tf.nn.embedding_lookup(self.W,
                                                   self.input_target_sent)
        self.target_label = tf.placeholder(tf.float32,
                                           [None, None], 'target_label_one_hot')
        self.target_sent_mask = tf.placeholder(tf.float32,
                                                [None, sen_len],
                                                'target_label')

        self.input_target_sent_test = tf.placeholder(tf.int32,
                                                      [None, sen_len],
                                                      'target_sent_test')
        self.target_sent_test = tf.nn.embedding_lookup(self.W,
                                                        self.input_target_sent_test)
        self.target_sent_mask_test = tf.placeholder(tf.float32,
                                                     [None, sen_len],
                                                     'target_sent_mask_test')

        self.ss_encoded_sents_avg_test = tf.placeholder(tf.float32,
                                                         [None, None, None],
                                                         'support_set_emb_test')

        self.input_ood_sents = tf.placeholder(tf.int32,
                                               [None, self.params['ood_example_size'],
                                                sen_len],
                                               'input_ood_sents')
        self.ood_sents = tf.nn.embedding_lookup(self.W,
                                                 self.input_ood_sents)
        self.ood_sents_mask = tf.placeholder(tf.int32,
                                              [None, self.params['ood_example_size'],
                                               sen_len],
                                              'ood_sents_mask')

        self.is_training = tf.placeholder(tf.bool, name='training-flag')

    def get_prototype(self, ss_presententations, ss_labels):
        label_level_sum_embs = tf.matmul(ss_labels, ss_presententations,
                                         transpose_a=True)
        label_level_sum_cnt = tf.reduce_sum(ss_labels, axis=1)
        protos = label_level_sum_embs / tf.expand_dims(label_level_sum_cnt, axis=2)
        return protos

    def build(self):
        with tf.name_scope("losses"):
            encoded_sents = self.ss_g(sent_input=self.support_set_sents,
                                       support_set=None,
                                       is_training=self.is_training,
                                       reuse=False,
                                       mask=self.support_set_sents_mask)

            shape_list_after_g = tf.shape(encoded_sents)
            # shape: B, SS, E
            encoded_sents = tf.reshape(encoded_sents,
                                        [-1,
                                         tf.shape(self.support_set_sents)[1],
                                         shape_list_after_g[-1]])

            target_encoded_sents = self.ss_g(sent_input=self.target_sent,
                                              support_set=None,
                                              is_training=self.is_training,
                                              reuse=True,
                                              mask=self.target_sent_mask)  # shape: batch_size, emb

            ood_encoded_sents = self.ss_g(sent_input=self.ood_sents,
                                           support_set=None,
                                           is_training=self.is_training,
                                           reuse=True,
                                           mask=self.ood_sents_mask)
            ood_shape_list_after_g = tf.shape(ood_encoded_sents)
            # ood_encoded_sents(B, OOD, E)
            ood_encoded_sents = tf.reshape(ood_encoded_sents,
                                            [-1, tf.shape(self.ood_sents)[1],
                                             ood_shape_list_after_g[-1]])

            # encoded_prototype.shape(B, Y, E)
            encoded_prototype = self.get_prototype(encoded_sents, self.support_set_labels)

            # loss-1
            similarities_label_level = self.dn(support_set=encoded_prototype,
                                               input_sent=target_encoded_sents,
                                               name="distance_calculation")
            softmax_similarities_proto = self.classify(similarities_label_level,
                                                       name='classify',
                                                       softmax_factor=self.params['softmax_factor'])

            logloss_ontopic = -tf.log(tf.reduce_sum(self.target_label 
                                      * softmax_similarities_proto,
                                      reduction_indices=[1]))

            # loss-2
            similarities_label_level01 = (similarities_label_level + 1.0) / 2.0
            tmp = tf.reduce_sum(self.target_label * similarities_label_level01, reduction_indices=[1])
            loss_target_gt_distance = tf.maximum(0.0, self.params['ood_threshold']
                                                 + self.params['ood_threshold_margin'] 
                                                 - tmp)

            # loss-3
            # ood_similarities_label_level(B, OOD, Y)
            ood_similarities_label_level = self.dn(support_set=encoded_prototype,
                                                   input_sent=ood_encoded_sents,
                                                   name="ood_distance_calculation")
            ood_similarities_label_level01 = (ood_similarities_label_level + 1.0) / 2.0
            ood_similarities_label_level_max01 = tf.reduce_max(ood_similarities_label_level01, axis=2)
            ood_hinge_loss = tf.reduce_mean(tf.maximum(0.0, ood_similarities_label_level_max01
                                            - self.params['ood_threshold']
                                            + self.params['ood_threshold_margin']), axis=-1)
            logloss_ontopic = tf.reduce_mean(logloss_ontopic)
            loss_target_gt_distance = tf.reduce_mean(loss_target_gt_distance)
            ood_hinge_loss = tf.reduce_mean(ood_hinge_loss)
            crossentropy_loss = self.params['alpha_indomain'] * logloss_ontopic \
                + self.params['alpha_pos'] * loss_target_gt_distance \
                + self.params['alpha_neg'] * ood_hinge_loss

            target_encoded_sents_test = self.ss_g(sent_input=self.target_sent_test,
                                                   support_set=None,
                                                   is_training=self.is_training,
                                                   reuse=True,
                                                   mask=self.target_sent_mask_test)  # shape: batch_size, emb

            similarities_avg_level_test = self.dn(support_set=self.ss_encoded_sents_avg_test,
                                                  input_sent=target_encoded_sents_test,
                                                  name="distance_calculation_test")

            softmax_similarities_proto_test = self.classify(similarities_avg_level_test,
                                                            name='classify',
                                                            softmax_factor=self.params['softmax_factor'])
            similarities_avg_level_test_01 = (similarities_avg_level_test + 1.0) / 2.0

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return crossentropy_loss, \
            softmax_similarities_proto_test, \
            similarities_avg_level_test_01, \
            encoded_prototype, \
            logloss_ontopic, \
            loss_target_gt_distance, \
            ood_hinge_loss

    def get_train_op(self, loss):
        print('current_learning_rate', self.params['learning_rate'])

        c_opt = tf.train.AdamOptimizer(beta1=0.9,
                                       learning_rate=self.params['learning_rate'])

        if self.params['enable_batchnorm'] is True:
            v1 = tf.get_default_graph().get_tensor_by_name("g/BatchNorm/moving_mean:0")
            v2 = tf.get_default_graph().get_tensor_by_name("g/BatchNorm/moving_variance:0")
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, v1)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, v2)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_variables = self.variables + [self.W]
            for variable in tf.global_variables():
                print("tf.global_variables()-->", variable)
            c_error_opt_op = c_opt.minimize(loss)
        return c_error_opt_op

    def init(self):
        loss, \
           softmax_similarities_proto_test, \
           similarities_avg_level_test_01, \
           encoded_prototype, \
           logloss_ontopic, \
           loss_target_gt_distance, \
           ood_hinge_loss = self.build()

        train_op = self.get_train_op(loss)
        self.train_op = train_op
        self.loss = loss
        self.encoded_prototype = encoded_prototype
        self.test_preds = softmax_similarities_proto_test
        self.test_preds_unnorm = similarities_avg_level_test_01
        self.loss_ontopic = logloss_ontopic
        self.loss_target_gt_distance = loss_target_gt_distance
        self.loss_ood_hinge = ood_hinge_loss
