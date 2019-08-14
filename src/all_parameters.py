from __future__ import print_function


def str2bool(v):
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        raise Exception('bad string for boolean')


def set_default_values():
    '''set default values for every possible parameters.'''
    params = dict()

    params['total_epoch'] = 5000
    params['single_workspace'] = False

    # data required
    params['w2vfile'] = '/u/mingtan/data/few-shot-learning/word2vectors/simple.token.vectors'
    params['mnet_training_dir'] = ''
    params['mnet_training_list'] = ''
    params['mnet_dev_dir'] = ''
    params['mnet_dev_list'] = ''

    params['testing_dir'] = ''
    params['testing_list'] = ''

    params['model_string'] = 'dummy'
    params['model_dir'] = '/dccstor/slad/mingtan/offtopic_paper/outputs/models'

    # parameters that determine the shape of the input
    params['batch_size'] = 10
    params['max_length'] = 40
    params['emb_size'] = 100
    params['min_ss_size'] = 200

    # parameters about the model
    params['hidden_size'] = 100

    # model tricks
    params['softmax_factor'] = 5.0
    params['learning_rate'] = 0.001
    params['remove_target_from_support_set'] = True
    params['dropout_keep_prob'] = 1.0

    # loaded in main function
    params['vocabulary'] = None
    params["wordvectors"] = None

    params['offtopic_label'] = 'OUT_OF_DOMAIN_LABEL'

    params['sampling_classes'] = 0
    params['topk_ss'] = 0
    params['layer_num'] = 1

    params['enable_batchnorm'] = True
    params['mtl_num_tasks'] = 0
    params['filter_size'] = 3
    params['ood_threshold'] = 0.6
    params['ood_threshold_margin'] = 0.2
    params['ood_example_size'] = 5
    params['alpha_pos'] = 1.0
    params['alpha_neg'] = 1.0
    params['alpha_indomain'] = 1.0
    params['seed'] = 3143

    return params


def get_all_parameters(config_parser=None, section=None, print_params=True):
    '''read all parameters'''
    params = set_default_values()
    if not section or not config_parser:
        print('Use all default parameters.')
        return params

    try:
        options = config_parser.options(section)
        for option in options:
            if option not in params:
                Exception("Bad or missing entry is configuration file")
            paramstr = config_parser.get(section, option)
            if paramstr == 'True' or paramstr == 'False':
                params[option] = str2bool(paramstr)
            else:
                params[option] = paramstr

        params['total_epoch'] = int(params['total_epoch'])
        params['batch_size'] = int(params['batch_size'])
        params['max_length'] = int(params['max_length'])
        params['hidden_size'] = int(params['hidden_size'])
        params['softmax_factor'] = float(params['softmax_factor'])
        params['learning_rate'] = float(params['learning_rate'])
        params['sampling_classes'] = int(params['sampling_classes'])
        params['min_ss_size'] = int(params['min_ss_size'])
        params['emb_size'] = int(params['emb_size'])
        params['dropout_keep_prob'] = float(params['dropout_keep_prob'])
        params['topk_ss'] = int(params['topk_ss'])
        params['layer_num'] = int(params['layer_num'])
        params['mtl_num_tasks'] = int(params['mtl_num_tasks'])
        params['filter_size'] = int(params['filter_size'])
        params['ood_threshold'] = float(params['ood_threshold'])
        params['ood_threshold_margin'] = float(params['ood_threshold_margin'])
        params['ood_example_size'] = int(params['ood_example_size'])
        params['alpha_pos'] = float(params['alpha_pos'])
        params['alpha_neg'] = float(params['alpha_neg'])
        params['alpha_indomain'] = float(params['alpha_indomain'])
        params['seed'] = int(params['seed'])
    except Exception as e:
        print(e.message)
        exit(1)

    if print_params:
        print_parameters(params)
    return params


def print_parameters(params):
    '''print parameters'''
    print("All parameters:")
    for p in params:
        print("\t{}==>{}".format(p, params[p]))
