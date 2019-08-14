from __future__ import print_function
import json
import tensorflow as tf
import numpy
from all_parameters import get_all_parameters


class wordvector_integrator:
    def __init__(self, params):
        self.params = params
        voc_info = json.load(open(self.params['initial_manager_file'], 'r'))
        self.idx2str = voc_info['idx2str']
        self.str2idx = voc_info['str2idx']

        self.str2idx = dict([(key, int(self.str2idx[key])) for key in self.str2idx])
        self.idx2str = dict([(int(key), self.idx2str[key]) for key in self.idx2str])

        self.initial_word_vector_tensor = tf.get_variable("embedding/embedding",
                                                          shape=[len(self.idx2str), 100],
                                                          dtype=tf.float32)
        self.initial_w2v = None

    def load_initial_wordvectors(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, self.params['initial_model_prefix_emb'])
        WV = sess.run([self.initial_word_vector_tensor])
        self.initial_w2v = WV[0]
        print("initial word vector length", len(self.initial_w2v))
        print("initial vocabulary size", len(self.str2idx))

    def combine_with_indomain(self, voclist):
        neww2v = self.initial_w2v.tolist()
        for wd in voclist:
            if wd not in self.str2idx:
                # self.initial_w2v = numpy.concatenate((self.initial_w2v,
                # [(numpy.random.rand(self.params['emb_size']) - 0.5) *2]))
                neww2v.append(((numpy.random.rand(self.params['emb_size']) - 0.5) * 2).tolist())

                self.str2idx[wd] = len(self.str2idx)
                self.idx2str[len(self.idx2str)] = wd
        self.initial_w2v = numpy.array(neww2v)
        return self.str2idx, self.initial_w2v

