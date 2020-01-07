# coding: utf-8

import os
import json
import random
import numpy as np
import h5py


class Dataloader(object):
    '''
    Usage: 
        dataloader = Dataloader(config, 'train')
        batch_data = dataloader.batch_data_iterator()
    '''

    def __init__(self, config, split):
        self.config = config
        self.split = split

        assert self.split in {'train', 'val', 'test'}
        self.split_gt_dict = json.load(open(os.path.join(self.config.split_gt_info_path, self.split + '_gt.json'), 'r'))
        self.video_names = self.split_gt_dict.keys()

        self.split_size = len(self.video_names)

        self.feat_rgb = h5py.File(self.config.feat_path_rgb, 'r')
        self.feat_flow = h5py.File(self.config.feat_path_flow, 'r')

        self.gt_scores = self.generate_gt_scores()

    def calc_ioa(self, gt_starts, gt_ends, anchor_start, anchor_end):
        '''
        calc intersection over anchor length, frame_level
        gt_starts, gt_ends: multi values, shape: [gt_num]
        anchor_start, anchor_end: single value
        '''
        intersection_starts = np.maximum(gt_starts, anchor_start)
        intersection_ends = np.minimum(gt_ends, anchor_end)
        intersection_lens = np.maximum(intersection_ends - intersection_starts, 0)
        anchor_len = anchor_end - anchor_start
        ioa = intersection_lens / float(anchor_len)
        return ioa

    def generate_gt_scores(self):
        gt_scores = {}
        for video_name in self.video_names:
            gt_frame_stamps = np.asarray(self.split_gt_dict[video_name]['gt_frame_stamps'])
            cur_video_feat_len = self.split_gt_dict[video_name]['valid_frame_num'] / self.config.feat_resolution

            # shape: [gt_num]
            gt_frame_starts = gt_frame_stamps[:, 0]
            gt_frame_ends = gt_frame_stamps[:, 1]
            gt_frame_lens = gt_frame_ends - gt_frame_starts
            gt_boundary_region_lens = np.maximum(gt_frame_lens / 10, self.config.feat_resolution)

            # shape: [gt_num, 2]
            gt_start_regions = np.stack(
                (gt_frame_starts - gt_boundary_region_lens / 2, gt_frame_starts + gt_boundary_region_lens / 2), axis=1)
            gt_end_regions = np.stack(
                (gt_frame_ends - gt_boundary_region_lens / 2, gt_frame_ends + gt_boundary_region_lens / 2), axis=1)

            # shape: [cur_video_feat_len]
            start_scores = []
            for i in range(cur_video_feat_len):
                feat_frame_start = i * self.config.feat_resolution
                feat_frame_end = (i + 1) * self.config.feat_resolution
                best_score = np.max(
                    self.calc_ioa(gt_start_regions[:, 0], gt_start_regions[:, 1], feat_frame_start, feat_frame_end))
                start_scores.append(best_score)

            end_scores = []
            for i in range(cur_video_feat_len):
                feat_frame_start = i * self.config.feat_resolution
                feat_frame_end = (i + 1) * self.config.feat_resolution
                best_score = np.max(
                    self.calc_ioa(gt_end_regions[:, 0], gt_end_regions[:, 1], feat_frame_start, feat_frame_end))
                end_scores.append(best_score)

            action_scores = []
            for i in range(cur_video_feat_len):
                feat_frame_start = i * self.config.feat_resolution
                feat_frame_end = (i + 1) * self.config.feat_resolution
                best_score = np.max(self.calc_ioa(gt_frame_starts, gt_frame_ends, feat_frame_start, feat_frame_end))

                action_scores.append(best_score)

            cur_gt_scores = {'start_scores': start_scores, 'end_scores': end_scores, 'action_scores': action_scores}
            gt_scores[video_name] = cur_gt_scores
        return gt_scores

    def mask_process(self, batch_feat, batch_score):
        '''
        input:
            batch_feat:  [batch_size, window_size(vary), feat_dimension]
            batch_score: [batch_size, window_size(vary), recap_length*3]
        return:
            batch_feat_masked, batch_score_masked
            batch_mask: [batch_size, window_size(max)], np.int32
        '''
        window_sizes = []
        for video_feat in batch_feat:
            cur_window_size = video_feat.shape[0]
            window_sizes.append(cur_window_size)
        max_window_size = max(window_sizes)
        # print window_sizes

        batch_feat_masked = np.zeros((len(batch_feat), max_window_size, batch_feat[0].shape[-1]), dtype=np.float32)
        batch_score_masked = np.zeros((len(batch_feat), max_window_size, batch_score[0].shape[-1]), dtype=np.float32)
        batch_mask = np.zeros((len(batch_feat), max_window_size), dtype=np.int32)

        for i in range(len(batch_feat)):
            batch_feat_masked[i, :window_sizes[i], :] = batch_feat[i]
            batch_score_masked[i, :window_sizes[i], :] = batch_score[i]
            batch_mask[i, :window_sizes[i]] = 1

        return batch_feat_masked, batch_score_masked, batch_mask

    def balance_mask_func(self, score, ratio=[1, 1, 1]):
        '''
        score: start + end + action masked_scores, shape: [batch_size, window_size(masked), recap_length]
        '''
        start_scores = score[:, :, :self.config.recap_length]
        end_scores = score[:, :, self.config.recap_length: self.config.recap_length * 2]
        action_scores = score[:, :, self.config.recap_length * 2:]

        def balance_mask(score, cur_ratio):
            score_flat = np.reshape(score, [-1])
            thres_score = (score_flat >= 0.5).astype(np.float32)
            pos = np.where(thres_score == 1.)[0]
            neg = np.where(thres_score == 0.)[0]
            sample_idx = list(pos) + random.sample(neg, min(int(len(pos) * cur_ratio), len(neg)))
            mask = np.zeros_like(score_flat, dtype=np.float32)
            mask[sample_idx] = 1.
            mask = np.reshape(mask, score.shape)
            return mask

        balanced_start_mask = balance_mask(start_scores, ratio[0])
        balanced_end_mask = balance_mask(end_scores, ratio[1])
        balanced_action_mask = balance_mask(action_scores, ratio[2])

        return np.concatenate((balanced_start_mask, balanced_end_mask, balanced_action_mask), axis=-1)

    def batch_data_iterator(self):
        '''
        return:
            batch_data: dict.
        '''

        if self.split == 'train':
            random.shuffle(self.video_names)

        cur_ptr = 0
        while True:
            batch_feat = []  # shape: [batch_size, window_size(vary), feat_dimension]
            batch_score = []  # shape: [batch_size, window_size(vary), recap_length*3]
            batch_video_name = []  # shape: [batch_size]

            cur_batch_len = min(self.config.batch_size, self.split_size - cur_ptr)
            for video_idx in range(cur_batch_len):
                cur_video_name = self.video_names[video_idx + cur_ptr]
                batch_video_name.append(cur_video_name)

                cur_feat_rgb = self.feat_rgb[cur_video_name]['i3d_features']  # [whole_video_feat_len, 1024]
                cur_feat_flow = self.feat_flow[cur_video_name]['i3d_features']

                if self.config.feat_mode == 'rgb':

                    cur_feat = cur_feat_rgb
                elif self.config.feat_mode == 'flow':

                    cur_feat = cur_feat_flow
                else:
                    align_len = cur_feat_flow.shape[0]
                    cur_feat = np.concatenate((cur_feat_rgb[:align_len], cur_feat_flow[:align_len]), axis=-1)  # [whole_video_feat_len, feat_dimension]
                cur_feat_size = cur_feat.shape[0]

                assert cur_feat.shape[1] == self.config.feat_dimension, 'feat_dimension in config is {}, but the actual value read from the file is {}. They must match.'.format(
                    self.config.feat_dimension, cur_feat.shape[1])

                if self.split == 'train':
                    window_size = min(self.config.window_size, cur_feat_size)
                else:
                    window_size = cur_feat_size
                feat_start_idx = random.randint(0, cur_feat_size - window_size)
                feat_end_idx = feat_start_idx + window_size
                # shape: [window_size, feat_dimension]
                feat_sequence = cur_feat[feat_start_idx: feat_end_idx]

                batch_feat.append(feat_sequence)

                cur_scores = np.zeros((window_size, self.config.recap_length * 3))
                for i in range(feat_start_idx, feat_end_idx):
                    effective_cap_length = min(self.config.recap_length, i + 1)
                    for j in range(effective_cap_length):
                        cur_scores[i - feat_start_idx, j] = self.gt_scores[cur_video_name]['start_scores'][i - j]
                        cur_scores[i - feat_start_idx, j + self.config.recap_length] = \
                        self.gt_scores[cur_video_name]['end_scores'][i - j]
                        cur_scores[i - feat_start_idx, j + self.config.recap_length * 2] = \
                        self.gt_scores[cur_video_name]['action_scores'][i - j]

                batch_score.append(cur_scores)

            batch_feat_masked, batch_score_masked, batch_mask = self.mask_process(batch_feat, batch_score)
            balanced_score_mask = self.balance_mask_func(batch_score_masked, ratio=self.config.balance_ratio)

            batch_data = {'batch_feat_masked': batch_feat_masked, 'batch_score_masked': batch_score_masked,
                          'batch_mask': batch_mask, 'batch_balanced_mask': balanced_score_mask, 'batch_video_name': batch_video_name}

            yield batch_data

            cur_ptr += cur_batch_len
            if cur_ptr == self.split_size:
                cur_ptr = 0
                if self.split == 'train':
                    random.shuffle(self.video_names)

    @property
    def batch_num(self):
        return int(np.ceil(float(self.split_size) / self.config.batch_size))
