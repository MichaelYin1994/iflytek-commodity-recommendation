#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107291535
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(preprocessing_cudf_numba.py)对原始的*.txt数据进行预处理。不同的是，
本模块基于cuDF与numba进行高效的DataFrame的基础信息抽取与基于numba的高效数据预处理。
'''
import gc
import os
import sys
import warnings

import cudf
import numpy as np
from numba import njit
import seaborn as sns
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from utils import LoadSave

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2022
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)
###############################################################################

@njit
def njit_parse_target_id(str_list):
    '''解析字符串形式的targid的list，使用njit进行加速'''
    str_list = str_list[1:-1]
    str_list = str_list.split(',')

    return str_list


def parse_target_id(str_list):
    '''解析字符串形式的targid的list，返回int转换后的结果'''
    str_list_spllit = njit_parse_target_id(str_list)

    str_array = np.zeros((len(str_list_spllit), ))
    for i in range(len(str_array)):
        str_array[i] = int(str_list_spllit[i])

    return str_array


@njit
def njit_parse_time(str_list):
    '''解析字符串形式的time的list，使用njit加速'''
    str_list = str_list[1:-1]
    str_list = str_list.split(',')

    return str_list


def parse_time(str_list):
    '''解析字符串形式的time的list'''
    str_list_spllit = njit_parse_time(str_list)

    str_array = np.zeros((len(str_list_spllit), ))
    for i in range(len(str_array)):
        str_array[i] = int(np.float64(str_list_spllit[i]))

    return str_array


if __name__ == '__main__':
    # 读入原始的训练与测试数据
    # -------------------------
    start_time = time.time()

    NROWS = None
    IS_SAVE_DATA = False
    TRAIN_PATH = './data/train/'
    TEST_PATH = './data/test/'

    train_df = cudf.read_csv(
        TRAIN_PATH+'train.txt', header=None, nrows=NROWS,
        names=['pid', 'label',
               'gender', 'age',
               'targid', 'time',
               'province', 'city',
               'model', 'make'])
    test_df = cudf.read_csv(
        TEST_PATH+'apply_new.txt', header=None, nrows=NROWS,
        names=['pid', 'gender',
               'age', 'targid',
               'time', 'province',
               'city', 'model', 'make'])
    test_df['label'] = np.nan

    curr_time = time.time()
    print('[INFO] took {} loading end...'.format(
        np.round(curr_time - start_time, 10)))

    total_df = cudf.concat([train_df, test_df], axis=0, ignore_index=True)
    del train_df, test_df
    gc.collect()

    # 数据预处理与数据解析
    # -------------------------
    # 编码One-hot类型特征
    # *****************
    start_time = time.time()

    total_df['make'] = total_df['make'].str.split(' ').list.get(-1)

    for feat_name in ['province', 'city', 'model', 'make']:
        encoder = LabelEncoder()
        total_df[feat_name] = encoder.fit_transform(total_df[feat_name].to_array())

    curr_time = time.time()
    print('[INFO] took {} oht encoding end...'.format(
        np.round(curr_time - start_time, 10)))

    # 处理字符串类型特征
    # *****************
    start_time = time.time()

    total_targid_list = total_df['targid'].to_array().tolist()
    total_targid_list = list(map(parse_target_id, total_targid_list))

    total_timestamp_list = total_df['time'].to_array().tolist()
    total_timestamp_list = list(map(parse_time, total_timestamp_list))

    timestamp_argidx = [np.argsort(item) for item in total_timestamp_list]

    unmatch_idx = 0
    for i in range(len(total_targid_list)):
        if len(total_targid_list[i]) == len(timestamp_argidx[i]):
            total_targid_list[i] = np.array(total_targid_list[i])[timestamp_argidx[i]]
        else:
            total_targid_list[i] = np.array(total_targid_list[i])
            unmatch_idx += 1
    total_timestamp_list = [np.array(item)[sorted_idx] for item, sorted_idx in \
                            zip(total_timestamp_list, timestamp_argidx)]

    total_df.drop(['targid', 'time'], axis=1, inplace=True)
    total_df['targid_len'] = [len(item) for item in total_targid_list]

    curr_time = time.time()
    print('[INFO] took {} str processing end...'.format(
        np.round(curr_time - start_time, 10)))

    # 基础指标的抽取
    # -------------------------
    start_time = time.time()

    train_df = total_df[total_df['label'].notnull()].reset_index(drop=True)
    test_df = total_df[total_df['label'].isnull()].reset_index(drop=True)
    total_df.fillna(-1, inplace=True)

    curr_time = time.time()
    print('[INFO] took {} train test split processing...'.format(
        np.round(curr_time - start_time, 10)))

    # *****************
    start_time = time.time()

    for feat_name in ['gender', 'age', 'province', 'city', 'model', 'make']:
        tmp_val = train_df.groupby(feat_name)['label'].sum().values \
            / train_df.groupby(feat_name)['label'].count().values
        tmp_df = train_df.groupby(feat_name)['label'].count().reset_index()
        tmp_df['label_dist'] = tmp_val

        tmp_df = tmp_df.sort_values(
            by=['label_dist'], ascending=False)
        tmp_df.reset_index(inplace=True, drop=True)
        # print('++++++++++++')
        # print(tmp_df.iloc[:5])

    for feat_name in [['gender', 'age'], ['province', 'city'], ['model', 'make']]:
        tmp_val = train_df.groupby(feat_name)['label'].sum().values \
            / train_df.groupby(feat_name)['label'].count().values
        tmp_df = train_df.groupby(feat_name)['label'].count().reset_index()
        tmp_df['label_dist'] = tmp_val

        tmp_df = tmp_df.sort_values(
            by=['label_dist'], ascending=False)
        tmp_df.reset_index(inplace=True, drop=True)
        # print('++++++++++++')
        # print(tmp_df.iloc[:5])

    curr_time = time.time()
    print('[INFO] took {} groupby...'.format(
        np.round(curr_time - start_time, 10)))

    # 预处理数据的存储
    # -------------------------
    if IS_SAVE_DATA:
        file_processor = LoadSave(dir_name='./cached_data/')
        file_processor.save_data(
            file_name='total_df.pkl', data_file=total_df)
        file_processor.save_data(
            file_name='total_targid_list.pkl', data_file=total_targid_list)
        file_processor.save_data(
            file_name='total_timestamp_list.pkl', data_file=total_timestamp_list)
