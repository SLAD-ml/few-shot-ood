import sys
import math
from workspace import workspace

def search_best_threshold(params, valid_output_info_list):
    dataset_best_thresholds = []
    dataset_best_values = []

    for valid_output_info in valid_output_info_list:
        bestT = 0
        bestV = 1
        best_frr = 0
        best_far = 1

        offsize = len([conf for pred, gt, conf in valid_output_info
                      if gt == params['offtopic_label']])
        insize = len([conf for pred, gt, conf in valid_output_info
                     if gt != params['offtopic_label']])

        print('offsize, insize', offsize, insize)
        sorted_valid_output_info = sorted(valid_output_info, key=lambda x: x[2])

        accepted_oo = offsize
        rejected_in = 0.0
        threshold = 0.0
        ind = 0
        for pred, gt, conf in sorted_valid_output_info[:-1]:
            threshold = (sorted_valid_output_info[ind][2] + 
                         sorted_valid_output_info[ind+1][2])/2.0
            if gt != params['offtopic_label']:
                rejected_in += 1.0
            else:
                accepted_oo -= 1.0

            frr = rejected_in / insize
            far = accepted_oo / offsize
            dist = math.fabs(frr - far)
            if dist < bestV:
                bestV = dist
                bestT = threshold
                best_frr = frr
                best_far = far
            ind += 1

        dataset_best_thresholds.append(bestT)
        dataset_best_values.append(bestV)
        print('bestT, bestV, bestFAR, bestFRR', 
              bestT, bestV, best_far, best_frr)

    return dataset_best_thresholds, dataset_best_values


def get_results(params, output_info, threshold):

    total_gt_ontopic_utt = len([gt for pred, gt, conf in output_info
                               if gt != params['offtopic_label']])
    total_gt_offtopic_utt = len(output_info) - total_gt_ontopic_utt

    accepted_oo = 0.0
    rejected_in = 0.0
    correct_domain_label = 0.0
    correct_wo_thr = 0.0
    correct_w_thr = 0.0

    for pred, gt, conf in output_info:
        if conf < threshold:
            pred1 = params['offtopic_label']
        else:
            pred1 = pred

        if gt == params['offtopic_label'] and pred1 != gt:
            accepted_oo += 1
        elif gt != params['offtopic_label'] and pred1 == params['offtopic_label']:
            rejected_in += 1
        else:
            correct_domain_label += 1

        if gt != params['offtopic_label'] and pred == gt:
            correct_wo_thr += 1
        if gt != params['offtopic_label'] and pred1 == gt:
            correct_w_thr += 1

    far = accepted_oo / total_gt_offtopic_utt
    frr = rejected_in / total_gt_ontopic_utt
    eer = 1 - correct_domain_label / len(output_info)
    ontopic_acc_ideal = correct_wo_thr / total_gt_ontopic_utt
    ontopic_acc = correct_w_thr / total_gt_ontopic_utt

    return eer, far, frr, ontopic_acc_ideal, ontopic_acc


def compute_values(params, experiment, result_data, epoch):

    t_macro_avg_eer = 0.0
    t_macro_avg_far = 0.0
    t_macro_avg_frr = 0.0

    t_macro_avg_acc_ideal = 0.0
    t_macro_avg_acc = 0.0

    test_output_info_list = []

    for workspace_idx in range(len(result_data)):
        curr_dev_workspace = result_data[workspace_idx]
        _, _, test_output_info = \
            experiment.run_testing_epoch(epoch=epoch,
                                         test_workspace=curr_dev_workspace)
        test_output_info_list.append(test_output_info)

    thesholds, _ = search_best_threshold(params, test_output_info_list)

    for workspace_idx in range(len(result_data)):
        curr_dev_workspace = result_data[workspace_idx]
        print(curr_dev_workspace.target_sets_files[2])

        test_output_info = test_output_info_list[workspace_idx]
        test_eer, test_far, test_frr, test_ontopic_acc_ideal, \
            test_ontopic_acc = get_results(params, test_output_info, 
                                           thesholds[workspace_idx])
        print('test(eer, far, frr, ontopic_acc_ideal, ontopic_acc) %.3f, %.3f, %.3f, %.3f, %.3f' %
              (test_eer, test_far, test_frr,
               test_ontopic_acc_ideal,
               test_ontopic_acc))

        t_macro_avg_eer += test_eer
        t_macro_avg_far += test_far
        t_macro_avg_frr += test_frr
        t_macro_avg_acc_ideal += test_ontopic_acc_ideal
        t_macro_avg_acc += test_ontopic_acc

    t_macro_avg_eer /= len(result_data)
    t_macro_avg_far /= len(result_data)
    t_macro_avg_frr /= len(result_data)
    t_macro_avg_acc_ideal /= len(result_data)
    t_macro_avg_acc /= len(result_data)
    return t_macro_avg_eer, t_macro_avg_far, t_macro_avg_frr, \
        t_macro_avg_acc_ideal, t_macro_avg_acc, test_output_info_list


def get_data(params, file_list, role):
    workspaces = []
    with open(file_list) as fi:
        i = 0
        for wid in fi:
            wid = wid.strip().split('\t')[0]
            workspaces.append(workspace(wid, params, role))
            print('get_data:', i)
            sys.stdout.flush()
            i += 1
    return workspaces
