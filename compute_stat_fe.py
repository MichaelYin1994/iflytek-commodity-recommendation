#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107170104
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(compute_stat_fe.py)进行统计特征工程。
'''

import gc
import os
import sys
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder

from utils import LoadSave

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2022
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)
###############################################################################

if __name__ == '__main__':
    # 读入原始的训练与测试数据
    # -------------------------
    file_processor = LoadSave(dir_name='./cached_data/')
    total_df = file_processor.load_data(file_name='total_df.pkl')
    total_df.fillna(-1, axis=1, inplace=True)

    total_feat_mat = None

    # 转csr矩阵形式进行处理
    # -------------------------
    feat_name_list = ['gender']
    for feat_name in feat_name_list:
        encoder = OneHotEncoder()
        tmp_sp_mat = encoder.fit_transform(
            total_df[feat_name].values.reshape(-1, 1))

        if total_feat_mat is None:
            total_feat_mat = tmp_sp_mat
        else:
            total_feat_mat = hstack([total_feat_mat, tmp_sp_mat]).tocsr()

    feat_name_list = ['age', 'targid_len']
    for feat_name in feat_name_list:
        tmp_sp_mat = csr_matrix(total_df[feat_name].values.reshape(-1, 1))
        total_feat_mat = hstack([total_feat_mat, tmp_sp_mat]).tocsr()

    # 存储统计特征工程结果
    # -------------------------
    file_processor.save_data(
        file_name='total_sp_stat_mat.pkl', data_file=total_feat_mat)
