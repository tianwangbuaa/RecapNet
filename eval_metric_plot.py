# coding: utf-8
"""
Evaluate performance of generated proposals: avg recall vs avg proposal number, recall@1000 proposals vs tiou threshold

This scrip is directly modified from Victor Escorcia's code
"""

import os
import numpy as np
import json
import h5py
import math
import matplotlib.pyplot as plt
import time


def segment_tiou(target_segments, test_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments(ground truth) : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments(proposal) : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [m x n] with IOU ratio.
    Note: It assumes that target-segments are more scarce than test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    tiou = np.empty((m, n))
    for i in xrange(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0]) + (
                    target_segments[i, 1] - target_segments[i, 0]) - intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        tiou[i, :] = intersection / union
    return tiou


def recall_vs_tiou_thresholds(proposals, ground_truth, nr_proposals=1000, tiou_thresholds=np.arange(0.05, 1.05, 0.05)):
    """ Computes recall at different tiou thresholds given a fixed
        average number of proposals per video.

    Parameters
    ----------
    proposals: Dictionary
        each video id corresponds to a list
        each list contains items formatting ['timestamp': [start, end], 'score': score]

    ground_truth: Dictionary
        like proposals

    tiou_thresholds : 1darray, optional
        array with tiou threholds.

    Outputs
    -------
    average_recall : 1darray
        recall averaged over a list of tiou threshold.
    proposals_per_video : 1darray
        average number of proposals per video.
    """
    # Get list of videos.
    # video_lst = proposals['video-name'].unique()
    video_lst = np.array(list(ground_truth.keys()))

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    for videoid in video_lst:
        # Get proposals for this video.
        # prop_idx = proposals['video-name'] == videoid
        # this_video_proposals = proposals[prop_idx][['f-init', 'f-end']].values
        this_video_proposals = np.array([item['timestamp'] for item in proposals[videoid]], dtype='float32')

        # Sort proposals by score.
        # sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        # this_video_proposals = this_video_proposals[sort_idx, :]
        this_proposal_scores = np.array([item['score'] for item in proposals[videoid]])
        sort_idx = this_proposal_scores.argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        # Get ground-truth instances associated to this video.
        # gt_idx = ground_truth['video-name'] == videoid
        # this_video_ground_truth = ground_truth[gt_idx][['f-init', 'f-end']].values
        this_video_ground_truth = np.array(ground_truth[videoid]['gt_second_stamps'], dtype='float32')

        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)

    # To obtain the average number of proposals, we need to define a
    # percentage of proposals to get per video.
    # see #line228
    proposal_total_number = 0
    for videoid in proposals.keys():
        proposal_total_number += len(proposals[videoid])
    pcn = (video_lst.shape[0] * float(nr_proposals)) / float(proposal_total_number)

    # Computes recall at different tiou thresholds.
    matches = np.empty((video_lst.shape[0], tiou_thresholds.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty(tiou_thresholds.shape[0])
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):
        print('running for tiou = %f' % tiou),
        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]

            # Get number of proposals at the fixed percentage of total retrieved.
            nr_proposals = int(score.shape[1] * pcn)
            # Find proposals that satisfies minimum tiou threhold.
            matches[i, ridx] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) > 0).sum()

        # Computes recall given the set of matches per video.
        recall[ridx] = matches[:, ridx].sum(axis=0) / positives.sum()
        print(':%f' % recall[ridx])

    return recall, tiou_thresholds


def average_recall_vs_nr_proposals(proposals, ground_truth, tiou_thresholds=np.linspace(0.5, 1.0, 11)):
    """ Computes the average recall given an average number
        of proposals per video.

    Parameters
    ----------
    proposals: Dictionary
        each video id corresponds to a list
        each list contains items formatting ['timestamp': [start, end], 'score': score]

    ground_truth: Dictionary
        like proposals

    tiou_thresholds : 1darray, optional
        array with tiou threholds.

    Outputs
    -------
    average_recall : 1darray
        recall averaged over a list of tiou threshold.
    proposals_per_video : 1darray
        average number of proposals per video.
    """
    # Get list of videos.
    # video_lst = proposals['video-name'].unique()
    video_lst = np.array(list(ground_truth.keys()))

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    for videoid in video_lst:
        # Get proposals for this video.
        # shape: [X, 2]
        this_video_proposals = np.array([item['timestamp'] for item in proposals[videoid]], dtype='float32')

        # Sort proposals by score.
        # sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        # this_video_proposals = this_video_proposals[sort_idx, :]
        this_proposal_scores = np.array([item['score'] for item in proposals[videoid]])
        # 返回 index, 从大到小
        sort_idx = this_proposal_scores.argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        # Get ground-truth instances associated to this video.
        #
        this_video_ground_truth = np.array(ground_truth[videoid]['gt_second_stamps'], dtype='float32')

        # Compute tiou scores.
        # 第 m 行 n 列就是第 m 个 gt 分别与 n 个 prop 的 tiou
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)

    # Given that the length of the videos is really varied, we
    # compute the number of proposals in terms of a ratio of the total
    # proposals retrieved, i.e. average recall at a percentage of proposals
    # retrieved per video.

    # Computes average recall.
    tN = 1000
    pcn_lst = np.arange(1, tN + 1) / float(tN)
    matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])  # 每个 vid 有多少个 proposal
    recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):
        print('running for tiou = %f' % tiou)
        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_lst):
            # score: tiou, m * n
            # Total positives per video. (ground truth)
            positives[i] = score.shape[0]

            for j, pcn in enumerate(pcn_lst):
                # Get number of proposals as a percentage of total retrieved.
                nr_proposals = int(score.shape[1] * pcn)
                # Find proposals that satisfies minimum tiou threhold.
                # 在第ridx个阈值下， 第 i 个视频在前 j (百分比) 推荐下的 符合最低 tiou 阈值 的 gt 中被检到的数量
                matches[i, j] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) > 0).sum()

        # Computes recall given the set of matches per video.
        # 第 ridx 个阈值下的 recall
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # np.save('recall_data.npy', recall)

    # Recall is averaged.
    recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    # proposals_per_video = pcn_lst * (float(proposals.shape[0]) / video_lst.shape[0])
    proposal_total_number = 0
    for vid in proposals.keys():
        proposal_total_number += len(proposals[vid])
    proposals_per_video = pcn_lst * (float(proposal_total_number) / video_lst.shape[0])

    return recall, proposals_per_video

results = json.load(open('./prop.json', 'r'))

ground_truth_filename = './data/THUMOS14/split_gt_info/test_gt.json'
ground_truth = json.load(open(ground_truth_filename, 'r'))

recalls_avg, proposals_per_video = average_recall_vs_nr_proposals(results, ground_truth)

recalls_tiou, tiou_thresholds = recall_vs_tiou_thresholds(results, ground_truth, nr_proposals=100)

fid = h5py.File('recall_prop.hdf5', 'w')

fid.create_group('recall_avg').create_dataset('recalls_avg', data=recalls_avg)
fid.create_group('recall_tiou').create_dataset('recalls_tiou', data=recalls_tiou)
fid.create_group('proposals_per_video').create_dataset('proposal', data=proposals_per_video)
fid.close()

# save results for plot
fid2 = h5py.File('recapnet_result.hdf5', 'w')

fid2.create_group('AN-tIoU').create_dataset('recall_tiou', data=recalls_tiou)
fid2['AN-tIoU'].create_dataset('tiou_thresholds', data=np.arange(0.05, 1.05, 0.05))

fid2.create_group('AR-AN').create_dataset('average_nr_proposals', data=proposals_per_video)
fid2['AR-AN'].create_dataset('average_recall', data=recalls_avg)

fid2.close()

for recall, prop_per_video in zip(recalls_avg, proposals_per_video):
    if math.fabs(prop_per_video - 1000) <= 2:
        print('avg recall: %f, prop_per_video: %d' % (recall, prop_per_video))
    if math.fabs(prop_per_video - 500) <= 2:
        print('avg recall: %f, prop_per_video: %d' % (recall, prop_per_video))
    if math.fabs(prop_per_video - 400) <= 2:
        print('avg recall: %f, prop_per_video: %d' % (recall, prop_per_video))
    if math.fabs(prop_per_video - 300) <= 2:
        print('avg recall: %f, prop_per_video: %d' % (recall, prop_per_video))
    if math.fabs(prop_per_video - 200) <= 2:
        print('avg recall: %f, prop_per_video: %d' % (recall, prop_per_video))
    if math.fabs(prop_per_video - 100) <= 2:
        print('avg recall: %f, prop_per_video: %d' % (recall, prop_per_video))
    if math.fabs(prop_per_video - 50) <= 2:
        print('avg recall: %f, prop_per_video: %d' % (recall, prop_per_video))

print('Plotting avg recall vs avg proposal number ...')
# Define plot style.
method = {'legend': 'RecapNet', 'color': np.array([102, 166, 30]) / 255.0, 'marker': None,
          'linewidth': 4, 'linestyle': '-'}
fn_size = 14
plt.figure()

# Plots Average Recall vs Average number of proposals.
plt.semilogx(proposals_per_video, recalls_avg, label=method['legend'], color=method['color'],
             linewidth=method['linewidth'], linestyle=str(method['linestyle']))

plt.ylabel('Average Recall', fontsize=fn_size)
plt.xlabel('Average number of proposals', fontsize=fn_size)
plt.grid(b=True, which="both", linestyle='dotted')
plt.ylim([0, 1.0])
plt.xlim([10 ** 1, 4* 10 ** 3])
plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
plt.show()

# Plots recall at different tiou thresholds.
plt.plot(tiou_thresholds, recalls_tiou, label=method['legend'], color=method['color'], linewidth=method['linewidth'],
         linestyle=str(method['linestyle']))

plt.grid(b=True, which="both")
plt.ylabel('Recall@100 proposals', fontsize=fn_size)
plt.xlabel('tIoU', fontsize=fn_size)
plt.ylim([0, 1])
plt.xlim([0.1, 1])
plt.xticks(np.arange(0, 1.1, 0.1))
plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
plt.show()
