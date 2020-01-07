# coding: utf-8

from __future__ import print_function

import numpy as np
import h5py
import cv2
import os
from pprint import pprint
import time
import math

import tensorflow as tf
import i3d

import requests
import urllib

_START_TIME = time.time()

_IMAGE_SIZE = 224
_SAMPLE_VIDEO_FRAMES = None

_LOG_FILE = 'preprocess_output_rgb.txt'
log_file = open(_LOG_FILE, 'w')

_ANNOTATION_PATHS = ['./dataset/th14_temporal_annotations_test/annotation', './dataset/th14_temporal_annotations_validation/annotation']
raw_data_dir = ['./dataset/thumos/validation/', './dataset/thumos/TH14_test_set_mp4/']

annotated_vid_names = []
txt_path = []
for anno_path in _ANNOTATION_PATHS:
    txt_path.extend([os.path.join(anno_path, i) for i in os.listdir(anno_path)])
for txt_file in txt_path:
    txt_content = open(txt_file, 'r').readlines()
    txt_content = [line.strip().split(' ')[0] for line in txt_content]
    annotated_vid_names.extend(txt_content)
annotated_vid_names = list(set(annotated_vid_names))

annotated_vid_paths = [raw_data_dir[0] + vid + '.mp4' if 'validation' in vid else raw_data_dir[1] + vid + '.mp4' for vid in annotated_vid_names]
# pprint(annotated_vid_paths)
total_vid_cnt = len(annotated_vid_paths)

with tf.variable_scope('RGB'):
    rgb_input = tf.placeholder(tf.float32, shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    rgb_model = i3d.InceptionI3d(final_endpoint='Avg_pool_3d')
    avg, end_points = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_out = tf.squeeze(avg)
rgb_saver = tf.train.Saver(reshape=True)

f = h5py.File('thumos14_i3d_features_rgb_with_ucf101.hdf5', 'w')

with tf.Session() as sess:
    rgb_saver.restore(sess, './i3d-ucf101-rgb-flow-model/rgb.ckpt')

    for vid_index, vid_path in enumerate(annotated_vid_paths):
        start_time = time.time()
        vid_name = os.path.basename(vid_path).split('.')[0]
        cap = cv2.VideoCapture(vid_path)
        vid_frame_cnt = int(cap.get(7))  # 视频帧数
        valid_frame_cnt = vid_frame_cnt - vid_frame_cnt % 16
        g = f.create_group(vid_name)
        d1 = g.create_dataset('i3d_features', (valid_frame_cnt / 16, 1024))
        d2 = g.create_dataset('total_frames', (1,), dtype='i')
        d3 = g.create_dataset('valid_frames', (1,), dtype='i')
        d2[0] = vid_frame_cnt
        d3[0] = valid_frame_cnt

        for i in range(valid_frame_cnt / 16):
            stacked_frames = []
            for j in range(16):
                _, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stacked_frames.append(cv2.resize(frame, (_IMAGE_SIZE , _IMAGE_SIZE ), interpolation=cv2.INTER_AREA))
            stacked_frames = np.array(stacked_frames)
            stacked_frames = stacked_frames[np.newaxis, :]
            stacked_frames = (stacked_frames / 255. - 0.5) * 2.
            # HIGHLIGHT: 
            # Mixed_5c: i3d_feat shape: [2, 7, 7, 1024]
            # Avg_pool_3d: i3d_feat_shape: [?, 1024]  
            i3d_feat = sess.run([rgb_out], feed_dict={rgb_input: stacked_frames})
            d1[i] = i3d_feat
        cap.release()
        end_time = time.time()
        time_gap = end_time - start_time
        print('Processed "{}"[{}/{}], total_frame: {}, valid_frame: {}. used: {:.1f} s, total time: {} min'.format(vid_name, vid_index+1, total_vid_cnt, vid_frame_cnt, valid_frame_cnt, time_gap, int(math.ceil((end_time - _START_TIME) / 60))))
        print('Processed "{}"[{}/{}], total_frame: {}, valid_frame: {}. used: {:.1f} s, total time: {} min'.format(vid_name, vid_index+1, total_vid_cnt, vid_frame_cnt, valid_frame_cnt, time_gap, int(math.ceil((end_time - _START_TIME) / 60))), file=log_file)
f.close()

_END_TIME = time.time()
print('Finished! Used {} min to preprocess all these data.'.format(int((_END_TIME - _START_TIME) / 60)))
print('Finished! Used {} min to preprocess all these data.'.format(int((_END_TIME - _START_TIME) / 60)), file=log_file)
log_file.close()


