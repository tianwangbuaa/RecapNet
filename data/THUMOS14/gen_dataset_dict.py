# coding: utf-8

import json
import random
import os
from pprint import pprint

## some constants
feat_resolution = 16
train_val_split = 0.8

video_info_dict = json.load(open('./video_info.json', 'r')) 

anno_split_path = {'val': './annotation_txt/validation/annotation',
                   'test': './annotation_txt/test/annotation'}

if not os.path.exists('split_gt_info'):
    os.makedirs('split_gt_info')

for split_name in anno_split_path:
    print 'Processing {}:'.format(split_name)
    txt_paths = [os.path.join(anno_split_path[split_name], i) for i in os.listdir(anno_split_path[split_name])]

    output_dict = {}

    for txt_path in txt_paths:
        txt_content = open(txt_path).readlines()
        for line in txt_content:
            line = line.strip().split(' ')  # here two blanks
            video_name, start_time, end_time = line[0], float(line[2]), float(line[3])
            
            frame_num = video_info_dict[video_name]['frame_num']
            valid_frame_num = frame_num - frame_num % feat_resolution
            duration = video_info_dict[video_name]['duration']
            valid_duration = float(valid_frame_num) / frame_num * duration
            fps = video_info_dict[video_name]['frame_rate']  # float
            start_frame = int(start_time / duration * frame_num)
            end_frame = int(end_time / duration * frame_num)
            start_feat = start_frame / feat_resolution
            end_feat = end_frame / feat_resolution

            if not video_name in output_dict:
                assert video_name in video_info_dict
                output_dict[video_name] = {}
                output_dict[video_name]['frame_num'] = frame_num
                output_dict[video_name]['valid_frame_num'] = valid_frame_num
                output_dict[video_name]['duration'] = duration
                output_dict[video_name]['valid_duration'] = valid_duration
                output_dict[video_name]['fps'] = fps
                output_dict[video_name]['gt_frame_stamps'] = [[start_frame, end_frame]]
                output_dict[video_name]['gt_second_stamps'] = [[start_time, end_time]]
                output_dict[video_name]['gt_feat_stamps'] = [[start_feat, end_feat]]
                output_dict[video_name]['gt_num'] = len(output_dict[video_name]['gt_frame_stamps'])
            else:
                output_dict[video_name]['gt_frame_stamps'].append([start_frame, end_frame])
                output_dict[video_name]['gt_second_stamps'].append([start_time, end_time])
                output_dict[video_name]['gt_feat_stamps'].append([start_feat, end_feat])
                output_dict[video_name]['gt_num'] = len(output_dict[video_name]['gt_frame_stamps'])

    if split_name == 'val':
        val_keys = output_dict.keys()
        val_len = len(output_dict)
        split_train_num = int(val_len * train_val_split)
        split_val_num = val_len - split_train_num

        _seed = random.seed(1)
        random.shuffle(val_keys)
        split_train_keys = val_keys[:split_train_num]
        split_val_keys = val_keys[split_train_num:]

        output_train_dict = {k: output_dict[k] for k in split_train_keys}
        output_val_dict = {k: output_dict[k] for k in split_val_keys}

        print '{} split has {} videos. Now writting data...'.format('train', split_train_num)
        with open('./split_gt_info/train_gt.json', 'w') as f:
            json.dump(output_train_dict, f)
        
        print '{} split has {} videos. Now writting data...'.format('val', split_val_num)
        with open('./split_gt_info/val_gt.json', 'w') as f:
            json.dump(output_val_dict, f)
    else:
        # remove the falsely annotated video "270" during test.
        output_dict_ = {}
        for k, v in output_dict.items():
            if k == 'video_test_0000270':
                continue
            else:
                output_dict_[k] = v
        print '{} split has {} videos. Now writting data...'.format('test', len(output_dict))
        with open('./split_gt_info/test_gt.json', 'w') as f:
            json.dump(output_dict_, f)





