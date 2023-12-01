import os
import glob

import cv2
import numpy as np


def create_train_dataset(src_dir, dst_dir, class_dict={'b': 0, 'r': 1, 'w': 2, 'y': 3}):
    if src_dir.startswith('/'):
        src_dir = src_dir[:-1]
    dataset_name = os.path.split(src_dir)[-1]

    img_path_list = glob.glob(os.path.join(src_dir, '*/*/*.*'))

    f = open(os.path.join(dst_dir, 'train_list.txt'), 'w')
    for img_path in img_path_list:
        class_name = img_path.split('/')[-2]
        class_id = class_dict[class_name]
        img_rpath = img_path[len(src_dir) - len(dataset_name):]
        f.write('{} {}\n'.format(
            img_rpath,
            class_id
        ))
    f.close()


def create_val_dataset(src_dir, 
                       dst_dir, 
                       class_dict={
                           'lp_blue': 0,
                           'lp_red': 1,
                           'lp_white': 2,
                           'lp_yellow': 3
                       }):
    if src_dir.startswith('/'):
        src_dir = src_dir[:-1]
    dataset_name = os.path.split(src_dir)[-1]

    img_path_list = glob.glob(os.path.join(src_dir, '*/*/*/*.*'))

    f = open(os.path.join(dst_dir, 'val_list.txt'), 'w')
    for img_path in img_path_list:
        class_name = img_path.split('/')[-3]
        class_id = class_dict[class_name]
        img_rpath = img_path[len(src_dir) - len(dataset_name):]
        f.write('{} {}\n'.format(
            img_rpath,
            class_id
        ))
    f.close()


def create_test_dataset(src_dir, 
                       dst_dir, 
                       class_dict={'b': 0, 'r': 1, 'w': 2, 'y': 3}):
    if src_dir.startswith('/'):
        src_dir = src_dir[:-1]
    dataset_name = os.path.split(src_dir)[-1]

    with open(os.path.join(src_dir, 'gt.txt')) as f:
        content = f.readlines()

    f = open(os.path.join(dst_dir, 'test_list.txt'), 'w')
    for line in content:
        img_rpath, class_name = line.strip().split('\t')
        img_path = os.path.join(dataset_name, img_rpath)
        class_id = class_dict[class_name]
        f.write('{} {}\n'.format(
            img_path,
            class_id
        ))
    f.close()


def create_label_list(dst_dir, class_name_list=['blue', 'red', 'yellow', 'white']):
    with open(os.path.join(dst_dir, 'lpc_label_list.txt'), 'w') as f:
        for idx, class_name in enumerate(class_name_list):
            f.write('{} {}\n'.format(idx, class_name))


def split_and_concate_w_ratio(raw_img, ratio=1.00):
    i_h, i_w, _ = raw_img.shape
    if i_w / i_h < 2.5:
        img = raw_img.copy()
        if img.shape[0] % 2 == 1:
            img = np.concatenate((img, img[-1:]), axis=0)
        top_limit = int((img.shape[0] // 2) * ratio)
        bottom_limit = img.shape[0] - top_limit
        img = np.concatenate((img[:top_limit], img[bottom_limit:]), axis=1)
        return img
    else:
        return raw_img


def split_img(img_template):
    img_path_list = glob.glob(img_template)
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        img = split_and_concate_w_ratio(img)
        cv2.imwrite(img_path, img)


if __name__ == '__main__':
    # src_dir = 'dataset/lpc/lpc_data_v1'
    # dst_dir = 'dataset/lpc'
    # create_train_dataset(src_dir, dst_dir)

    # src_dir = 'dataset/lpc/red_yellow_white_blue'
    # dst_dir = 'dataset/lpc'
    # create_val_dataset(src_dir, dst_dir)

    # src_dir = 'dataset/lpc/lpc_20231112'
    # dst_dir = 'dataset/lpc'
    # create_test_dataset(src_dir, dst_dir)

    # dst_dir = 'dataset/lpc'
    # create_label_list(dst_dir)

    img_template = 'dataset/lpc_split/lpc_data_v1/*/*/*.*'
    split_img(img_template)

    img_template = 'dataset/lpc_split/red_yellow_white_blue/*/*/short/*.*'
    split_img(img_template)
