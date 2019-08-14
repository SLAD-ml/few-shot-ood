from __future__ import print_function
import numpy
import os
from simple_tokenizer import tokenizeSimple


def read_word_vectors(filename):
    wdmap = dict()
    W = []
    curr = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            items = line.split('\t')
            if len(items) == 2 and items[0] not in wdmap:
                wdmap[items[0]] = curr
                W.append([float(ii) for ii in items[1].split()])
                curr += 1
        f.close()
    print('wdmap', len(wdmap))
    print('len(W)', len(W))
    return wdmap, W


def read_word2idx(infile):
    word2idx = dict()
    with open(infile, 'r') as fi:
        for line in fi:
            wd, idx = line[:-1].split('\t')
            word2idx[wd] = int(idx)
        fi.close()
    return word2idx


def get_word_info(params):
    word2idx = dict()
    w2v = []
    word2idx, w2v = read_word_vectors(params['w2vfile'])
    enrich_word_info_with_train_file(word2idx,
                                     w2v,
                                     params['mnet_training_dir'],
                                     params['mnet_training_list'],
                                     params)
    print("After combined with train file")
    print("word2idx size:", len(word2idx))
    for i in range(len(w2v)):
        if len(w2v[i]) != params['emb_size']:
            raise Exception("wordvec idx %d has a dimension of %d" 
                            % (i, len(w2v[i])))
    w2v = numpy.array(w2v)
    return word2idx, w2v


def enrich_word_info_with_train_file(word2idx,
                                     w2v,
                                     mnet_training_wksp_dir,
                                     mnet_training_wksp_list, params):
    with open(mnet_training_wksp_list, 'r') as fi:
        wksp_ids = fi.readlines()
        wksp_ids = [wid.split('\t')[0] for wid in wksp_ids]
        if params['single_workspace'] is False:
            wksp_tr_files = [os.path.join(mnet_training_wksp_dir, 
                                          wksp.strip()+'.train')
                             for wksp in wksp_ids]
        else:
            wksp_tr_files = [os.path.join(mnet_training_wksp_dir,
                                          wksp.strip()+'.train')
                             for wksp in wksp_ids]
        fi.close()

    for wkspfile in wksp_tr_files:
        fi = open(wkspfile, 'r')
        for line in fi:
            line = line.strip()
            items = line.split('\t')
            if len(items) == 2:
                text, lb = items
            textwds = tokenizeSimple(text, params['max_length'])
            textids = []
            for wd in textwds:
                if wd in word2idx:
                    textids.append(word2idx[wd])
                else:
                    word2idx[wd] = len(word2idx)
                    if w2v is not None:
                        w2v.append(((numpy.random.rand(params['emb_size'])
                                   - 0.5) * 2).tolist())
    return word2idx, w2v


def write_word2idx(word2idx, outfile):
    with open(outfile, 'w') as fo:
        for wd in word2idx:
            fo.write(wd+'\t'+str(word2idx[wd])+'\n')
        fo.close()
    return
