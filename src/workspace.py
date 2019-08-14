from __future__ import print_function
import os
from simple_tokenizer import tokenizeSimple
import numpy
from random import randint
import random

# word2vec must start with </s> and <unk>
SE_INDEX = 0
UNKNOWN_WORD_INDEX = 1

SENT_WORDID = 0
SENT_LABELID = 1
SENT_WORD_MASK = 2
SENT_ORIGINAL_TXT = 3


class workspace:

    def __init__(self, workspace_name, params, role):
        ''' Description: '''
        self.role = role
        self.wid = workspace_name
        self.word2idx = params['vocabulary']
        self.supporting_sets = []
        self.flatten_supporting_sets = None
        self.lb2id = dict()
        self.lblist = []
        self.params = params

        self.target_sets = []
        self.target_sets_files = []
        self.labels_in_train = None
        self.train_intent2ids_list = None

        if self.role == 'train_workspace':
            trainfile = os.path.join(params['mnet_training_dir'],
                                     workspace_name+'.train')
            self.read_supporting_sets(trainfile)
            self.target_sets.append(self.read_sentences(trainfile))
            self.target_sets_files.append(trainfile)
            self.labels_in_train = range(0, len(self.supporting_sets))
            self.train_intent2ids_list = dict([(l, []) for l in self.labels_in_train])
            for idx in range(len(self.target_sets[0])):
                utt = self.target_sets[0][idx]
                label = utt[SENT_LABELID]
                self.train_intent2ids_list[label].append(idx)
        else:
            if self.role == 'valid_workspace':
                indir = params['mnet_dev_dir']
            elif self.role == 'test_workspace':
                indir = params['testing_dir']
            else:
                raise Exception('Unknown workspace role')

            support_set_file = os.path.join(indir,
                                            workspace_name+'.train')
            self.read_supporting_sets(support_set_file)
            self.labels_in_train = range(0, len(self.supporting_sets))

            trainfile = os.path.join(indir, workspace_name+'.train')
            if os.path.isfile(os.path.join(indir, workspace_name+'.dev')):
                validfile = os.path.join(indir, workspace_name+'.dev')
            else:
                validfile = os.path.join(indir, workspace_name+'.test')
            testfile = os.path.join(indir, workspace_name+'.test')

            self.target_sets.append(self.read_sentences(trainfile))
            self.target_sets.append(self.read_sentences(validfile))
            self.target_sets.append(self.read_sentences(testfile))
            self.target_sets_files.append(trainfile)
            self.target_sets_files.append(validfile)
            self.target_sets_files.append(testfile)

        self.create_flatten_supporting_set()

    def update_lbmaps(self, lb):
        if lb not in self.lb2id:
            newidx = len(self.lb2id)
            self.lb2id[lb] = newidx
            self.lblist.append(lb)

    def read_sentences(self, filename):
        target_info = []
        with open(filename, 'r') as fi:
            for line in fi:
                line = line.strip()
                items = line.split('\t')
                if len(items) > 1:
                    text, lb = items
                    self.update_lbmaps(lb)
                    textwds = tokenizeSimple(text, self.params['max_length'])
                    textids = []
                    for wd in textwds:
                        if wd in self.word2idx:
                            textids.append(self.word2idx[wd])
                        else:
                            textids.append(UNKNOWN_WORD_INDEX)
                    textids_mask = [int(wd == 0) * 0 + int(wd != 0) * 1
                                    for wd in textids]
                    target_info.append((textids, self.lb2id[lb],
                                        textids_mask, line))
                elif len(items) == 1:
                    pass
                else:
                    raise Exception("File Input Wrong")
        return target_info

    def read_supporting_sets(self, filename):
        sentence_data = self.read_sentences(filename)
        for idx in range(len(self.lblist)):
            self.supporting_sets.append([])
        for sent_info in sentence_data:
            assert(sent_info[1] is not None)
            self.supporting_sets[sent_info[1]].append(sent_info)

    def select_support_set(self, sentence_size_per_intent, 
                           target_group_id, target_sid, sampled_labels=None):

        sent_info = self.target_sets[target_group_id][target_sid]
        ss_sent_info = []

        for lb_idx in range(len(self.supporting_sets)):

            if self.params['sampling_classes'] > 1 and \
               lb_idx not in sampled_labels:
                continue

            find_remove_target = False
            if len(self.supporting_sets[lb_idx]) > 1 and \
               sent_info[SENT_LABELID] == lb_idx and \
               self.params['remove_target_from_support_set']:
                find_remove_target = True

            if len(self.supporting_sets[lb_idx]) > 0:
                randlist = numpy.random.permutation(range(
                                                    len(self.supporting_sets[lb_idx]))).tolist()
                selfidx = -1

                sentences_id_per_intent_tmp = randlist[:sentence_size_per_intent]
                sentences_id_per_intent = []
                if find_remove_target:
                    first_time = True
                    for idx in range(len(sentences_id_per_intent_tmp)):
                        a_support_info = self.supporting_sets[lb_idx][sentences_id_per_intent_tmp[idx]]
                        if sent_info[SENT_WORDID] == a_support_info[SENT_WORDID] and \
                           first_time is True:
                                first_time is False
                                selfidx = sentences_id_per_intent_tmp[idx]
                        else:
                            sentences_id_per_intent.append(sentences_id_per_intent_tmp[idx])
                else:
                    sentences_id_per_intent = sentences_id_per_intent_tmp

                while len(sentences_id_per_intent) < sentence_size_per_intent:
                    ll = len(randlist)
                    rd = randint(0, ll-1)
                    if (not find_remove_target) or selfidx != rd:
                        sentences_id_per_intent.append(randlist[rd])

                sentences_per_intent = [self.supporting_sets[lb_idx][i]
                                        for i in sentences_id_per_intent]
                ss_sent_info.extend(sentences_per_intent)
        return ss_sent_info

    def create_flatten_supporting_set(self):
        self.flatten_supporting_sets = []
        for lb_idx in range(len(self.supporting_sets)):
            for ss_info in self.supporting_sets[lb_idx]:
                self.flatten_supporting_sets.append(ss_info)

    def get_flatten_supporting_set(self):
        return self.flatten_supporting_sets

    def sample_classes(self, sampled_class_size):
        sampled_labels = random.sample(range(len(self.labels_in_train)),
                                       sampled_class_size)
        sampled_labels = [self.labels_in_train[ii] for ii in sampled_labels]
        return sampled_labels
