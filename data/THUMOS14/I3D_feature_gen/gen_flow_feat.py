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

_START_TIME = time.time()
log_file = open('preprocess_output_flow.txt', 'w')

flow_data_dir = './flow_data/'
all_video_names = os.listdir(flow_data_dir)
total_vid_cnt = len(all_video_names)

f = h5py.File('thumos14_i3d_features_flow.hdf5', 'w')

with tf.variable_scope('Flow'):
    flow_input = tf.placeholder(tf.float32, shape=(1, None, 224, 224, 2))
    flow_model = i3d.InceptionI3d(final_endpoint='Avg_pool_3d')
    flow_out, _ = flow_model(flow_input, is_training=False, dropout_keep_prob=1.0)
    flow_out = tf.squeeze(flow_out)
flow_saver = tf.train.Saver(reshape=True)

with tf.Session() as sess:
	flow_saver.restore(sess, './i3d-ucf101-rgb-flow-model/flow.ckpt')

	for vid_index, video_name in enumerate(all_video_names):
		start_time = time.time()
		flow_x_dir = os.path.join(flow_data_dir, video_name, 'flow_x')
		flow_y_dir = os.path.join(flow_data_dir, video_name, 'flow_y')

		flow_x_imgs = os.listdir(flow_x_dir)
		flow_x_imgs = sorted(flow_x_imgs, key=lambda x: int(x.split('.')[0].split('_')[-1]))

		vid_frame_cnt = len(flow_x_imgs)
		valid_frame_cnt = vid_frame_cnt - vid_frame_cnt % 16

		g = f.create_group(video_name)
        d1 = g.create_dataset('i3d_features', (valid_frame_cnt / 16, 1024))
        d2 = g.create_dataset('total_frames', (1,), dtype='i')
        d3 = g.create_dataset('valid_frames', (1,), dtype='i')
        d2[0] = vid_frame_cnt
        d3[0] = valid_frame_cnt

        for i in range(valid_frame_cnt / 16):
        	flow_array = np.zeros((1, 16, 224, 224, 2), dtype=np.float32)
        	for j in range(16):
        		flow_x_name = flow_x_imgs[i * 16 + j]
        		flow_y_name = flow_x_name.replace('x', 'y')
        		cur_flow_x_path = os.path.join(flow_x_dir, flow_x_name)
				cur_flow_y_path = os.path.join(flow_y_dir, flow_y_name)

				flow_array[0, j, :, :, 0] = cv2.imread(cur_flow_x_path).astype(np.float32)[:, :, 0] / 255. * 2 - 1
				flow_array[0, j, :, :, 1] = cv2.imread(cur_flow_y_path).astype(np.float32)[:, :, 0] / 255. * 2 - 1

			i3d_feat = sess.run(flow_out, feed_dict={flow_input: flow_array})
			d1[i] = i3d_feat

		end_time = time.time()
        time_gap = end_time - start_time

        print('Processed "{}"[{}/{}], total_frame: {}, valid_frame: {}. used: {:.1f} s, total time: {} min'.format(video_name, vid_index+1, total_vid_cnt, vid_frame_cnt, valid_frame_cnt, time_gap, int(math.ceil((end_time - _START_TIME) / 60))))
        print('Processed "{}"[{}/{}], total_frame: {}, valid_frame: {}. used: {:.1f} s, total time: {} min'.format(video_name, vid_index+1, total_vid_cnt, vid_frame_cnt, valid_frame_cnt, time_gap, int(math.ceil((end_time - _START_TIME) / 60))), file=log_file)

f.close()

_END_TIME = time.time()
print('Finished! Used {} min to preprocess all these data.'.format(int((_END_TIME - _START_TIME) / 60)))
print('Finished! Used {} min to preprocess all these data.'.format(int((_END_TIME - _START_TIME) / 60)), file=log_file)
log_file.close()

