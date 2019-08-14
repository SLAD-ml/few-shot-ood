from __future__ import print_function
from workspace import SENT_WORDID, SENT_LABELID, SENT_WORD_MASK, SENT_ORIGINAL_TXT
import numpy
import random
import os


class RunExperiment:

    def __init__(self, model, params, sess):
        self.model = model
        self.params = params
        self.sess = sess

    def get_train_batch(self, train_workspace, batch_size, all_ood_workspaces):
        sent_list = train_workspace.target_sets[0]
        nClasses_in_Train = len(train_workspace.labels_in_train)
        
        if self.params['sampling_classes'] <= 1 \
            or self.params['sampling_classes'] >= nClasses_in_Train:
                sampled_class_size = nClasses_in_Train
        else:
            sampled_class_size = self.params['sampling_classes']
        sampled_classes = train_workspace.sample_classes(sampled_class_size)
        sentence_size_per_intent = max(1, int(self.params['min_ss_size']
                                       / sampled_class_size))
        sent_id_batch = []
        for b in range(batch_size):
            selected_label = random.choice(sampled_classes)
            selected_utt = random.choice(train_workspace.train_intent2ids_list[selected_label])
            sent_id_batch.append(selected_utt)

        x_target_wid = [sent_list[i][SENT_WORDID] for i in sent_id_batch]
        y_target = [sent_list[i][SENT_LABELID] for i in sent_id_batch]
        x_target_mask = [sent_list[i][SENT_WORD_MASK] for i in sent_id_batch]

        x_support_set_wid = []
        y_support_set = []
        x_support_set_mask = []
        x_ood_wid = []
        x_ood_mask = []

        for target_sid in range(len(sent_id_batch)):
            selected_ood_sent_infos = []
            for _ in range(self.params['ood_example_size']):
                selected_ood_workspace = numpy.random.choice(all_ood_workspaces)
                fss = selected_ood_workspace.get_flatten_supporting_set()
                selected_id = numpy.random.choice([i for i in range(len(fss))])
                selected_ood_sent_info = fss[selected_id]
                selected_ood_sent_infos.append(selected_ood_sent_info)

            ss_sent_info = train_workspace.select_support_set(sentence_size_per_intent,
                                                              0,
                                                              sent_id_batch[target_sid],
                                                              sampled_classes)
            x_support_set_wid_per_sent = [sinfo[SENT_WORDID] for sinfo in ss_sent_info]
            y_support_set_per_sent = [sinfo[SENT_LABELID] for sinfo in ss_sent_info]
            x_support_set_mask_per_sent = [sinfo[SENT_WORD_MASK] for sinfo in ss_sent_info]

            x_support_set_mask.append(x_support_set_mask_per_sent)
            x_support_set_wid.append(x_support_set_wid_per_sent)
            y_support_set.append(y_support_set_per_sent)

            x_ood_wid_per_sent = [sinfo[SENT_WORDID] for sinfo in selected_ood_sent_infos]
            x_ood_mask_per_sent = [sinfo[SENT_WORD_MASK] for sinfo in selected_ood_sent_infos]
            x_ood_wid.append(x_ood_wid_per_sent)
            x_ood_mask.append(x_ood_mask_per_sent)

        return numpy.array(x_support_set_wid, dtype='int'), \
            numpy.array(y_support_set, dtype='int'), \
            numpy.array(x_support_set_mask, dtype='int'), \
            numpy.array(x_target_wid, dtype='int'), \
            numpy.array(y_target, dtype='int'), \
            numpy.array(x_target_mask, dtype='int'), \
            sampled_classes, \
            numpy.array(x_ood_wid, dtype='int'), \
            numpy.array(x_ood_mask, dtype='int')

    def run_training_epoch(self, train_workspace, all_ood_workspaces):

        assert(len(train_workspace.target_sets) == 1)
        batch_size = self.params['batch_size']
        x_support_set_wid, y_support_set, support_set_sents_mask, \
            x_target_wid, y_target, target_sent_mask, selected_labels, \
            x_neg_wid, neg_sent_mask = \
            self.get_train_batch(train_workspace, batch_size, all_ood_workspaces)

        y_support_set_one_hot = self.get_support_set_one_hot(y_support_set, selected_labels)
        y_target_one_hot = self.get_one_hot(y_target, selected_labels)
        model = self.model
        _, loss = self.sess.run(
                [model.train_op, model.loss],
                feed_dict={model.input_support_set_sents: x_support_set_wid,
                           model.support_set_sents_mask: support_set_sents_mask,
                           model.support_set_labels: y_support_set_one_hot,
                           model.input_target_sent: x_target_wid,
                           model.target_label: y_target_one_hot,
                           model.target_sent_mask: target_sent_mask,
                           model.ss_encoded_sents_avg_test: numpy.array([[[0]]]),
                           model.is_training: True,
                           model.input_ood_sents: x_neg_wid,
                           model.ood_sents_mask: neg_sent_mask
                           })
        return loss

    def get_supporting_set_embeddings(self, test_workspace):
        all_ss_info = test_workspace.get_flatten_supporting_set()

        txtbatch = [s[SENT_WORDID] for s in
                    all_ss_info]
        maskbatch = [s[SENT_WORD_MASK] for s in
                     all_ss_info]
        ybatch = [s[SENT_LABELID] for s in
                  all_ss_info]

        print('test_workspace.labels_in_train', len(test_workspace.labels_in_train))
        nClasses = len(test_workspace.labels_in_train)
        ybatch_one_shot = self.get_one_hot(ybatch, range(nClasses))

        center_emb_batch = self.sess.run(
                        self.model.encoded_prototype,
                        feed_dict={self.model.input_support_set_sents: [txtbatch],
                                   self.model.support_set_sents_mask: [maskbatch],
                                   self.model.support_set_labels: [ybatch_one_shot],
                                   self.model.is_training: False
                                   })
        return center_emb_batch

    def run_testing_epoch(self, epoch, test_workspace):
        nClasses = len(test_workspace.labels_in_train)
        avg_representations = self.get_supporting_set_embeddings(test_workspace)
        y_support_set_one_hot = self.get_one_hot(range(nClasses),
                                                 range(nClasses))
        print('avg_representations.shape', avg_representations.shape)
        rets_train = []
        rets_dev = []
        rets_test = []
        setidx = 0
        for target_set, target_set_file in zip(test_workspace.target_sets,
                                               test_workspace.target_sets_files):
            print('Testing: ', target_set_file)
            for target_sentence in target_set:
                text, label = target_sentence[SENT_ORIGINAL_TXT].split('\t')
                model = self.model
                preds_intent = self.sess.run(
                    model.test_preds_unnorm,
                    feed_dict={model.ss_encoded_sents_avg_test: avg_representations,
                               model.support_set_labels: [y_support_set_one_hot],
                               model.input_target_sent_test: [target_sentence[SENT_WORDID]],
                               model.target_sent_mask_test: [target_sentence[SENT_WORD_MASK]],
                               model.is_training: False})
                preds_intent = preds_intent[0]
                final_lb_intent_id = numpy.argmax(preds_intent)
                final_lb_intent = test_workspace.lblist[final_lb_intent_id]
                groundtruth_id = target_sentence[SENT_LABELID]
                groundtruth = test_workspace.lblist[groundtruth_id]
                conf = preds_intent[final_lb_intent_id]
                atuple = (final_lb_intent, groundtruth, conf)
                if setidx == 0:
                    rets_train.append(atuple)
                elif setidx == 1:
                    rets_dev.append(atuple)
                elif setidx == 2:
                    rets_test.append(atuple)
            setidx += 1

        return rets_train, rets_dev, rets_test

    def get_support_set_one_hot(self, support_set, classe_list):
        cls_id_map = dict()
        for lid in classe_list:
                    cls_id_map[lid] = len(cls_id_map)

        support_set_one_hot = numpy.zeros([len(support_set), 
                                          len(support_set[0]),
                                          len(cls_id_map)])
        for k in range(len(support_set)):
            for j in range(len(support_set[k])):
                support_set_one_hot[k][j][cls_id_map[support_set[k][j]]] = 1.0

        return support_set_one_hot

    def get_one_hot(self, y_target, classe_list):
        cls_id_map = dict()
        for lid in classe_list:
                    cls_id_map[lid] = len(cls_id_map)

        y_target_one_hot = numpy.zeros([len(y_target), len(cls_id_map)])
        for k in range(len(y_target)):
            y_target_one_hot[k][cls_id_map[y_target[k]]] = 1.0
        return y_target_one_hot

