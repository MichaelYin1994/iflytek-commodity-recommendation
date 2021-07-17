#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107171211
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(compute_embedding_fe.py)计算item embedding的特征工程。
'''

import gc
import os
import sys
import warnings
import multiprocessing as mp
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
from numba import njit
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder
from gensim.models import FastText, word2vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import LoadSave

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2022
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)
###############################################################################


@njit
def njit_compute_stat_feats(input_array=None):
    '''计算输入的array的一系列统计特征'''
    if len(input_array) == 1:
        return np.zeros((1, 5))
    stat_feats = np.zeros((1, 5))

    time_diff_array = input_array[1:] - input_array[:-1]
    stat_feats[0, 0] = np.mean(time_diff_array)
    stat_feats[0, 1] = np.std(time_diff_array)
    stat_feats[0, 2] = np.min(time_diff_array)
    stat_feats[0, 3] = np.max(time_diff_array)
    stat_feats[0, 4] = np.median(time_diff_array)

    return stat_feats


def comput_list_stat_feats(input_list=None):
    '''接口方法，将input_list转为array'''
    array_list = np.array(input_list)
    return njit_compute_stat_feats(array_list)


class GensimCallback(CallbackAny2Vec):
    '''计算每一个Epoch的词向量训练损失的回调函数。

    @Attributes:
    ----------
    epoch: {int-like}
    	当前的训练的epoch数目。
    verbose_round: {int-like}
    	每隔verbose_round轮次打印一次日志。
	loss: {list-like}
		保存每个epoch的Loss的数组。

    @References:
    ----------
    [1] https://stackoverflow.com/questions/54888490/gensim-word2vec-print-log-loss
    '''
    def __init__(self, verbose_round=3):
        self.epoch = 0
        self.loss = []

        if verbose_round == 0:
            verbose_round = 1
        self.verbose_round = verbose_round

    def on_epoch_end(self, model):
        '''在每个epoch结束的时候计算模型的Loss并且打印'''

        # 获取该轮的Loss值
        loss = model.get_latest_training_loss()
        self.loss.append(loss)

        if len(self.loss) == 1:
            pass
        else:
            loss_decreasing_precent = \
                (loss - self.loss[-2]) / self.loss[-2] * 100

            if divmod(self.epoch, self.verbose_round)[1] == 0:
                print('[{}]: word2vec loss: {:.2f}, decreasing {:.4f}%.'.format(
                    self.epoch, loss, loss_decreasing_precent))
        self.epoch += 1


def compute_sg_embedding(corpus=None, is_save_model=True,
                         model_name='skip_gram_model',
                         **kwargs):
    '''利用gensim的SKip-Gram模型训练并保存词向量。语料输入形式为：
        [['1', '2', '3'],
        ...,
        ['10', '23', '65', '9', '34']]
    '''
    print('\n[INFO] {} Skip-Gram embedding start.'.format(
        str(datetime.now())[:-4]))
    print('-------------------------------------------')
    model = word2vec.Word2Vec(corpus, sg=1,
                              workers=mp.cpu_count(),
                              compute_loss=True,
                              callbacks=[GensimCallback(verbose_round=3)],
                              **kwargs)
    print('-------------------------------------------')
    print('[INFO] {} Skip-Gram embedding end. \n'.format(
        str(datetime.now())[:-4]))

    # 保存Embedding模型
    # ---------------------------
    file_processor = LoadSave(
        dir_name='./pretraining_models/', verbose=1)

    if is_save_model:
        file_processor.save_data(
            file_name='{}.pkl'.format(model_name),
            data_file=model)
    return model


def compute_cbow_embedding(corpus=None, is_save_model=True,
                           model_name='cbow_model',
                           **kwargs):
    '''利用gensim的CBOW模型训练并保存词向量。语料输入形式为：
        [['1', '2', '3'],
        ...,
        ['10', '23', '65', '9', '34']]
    '''
    print('\n[INFO] CBOW embedding start at {}'.format(
        str(datetime.now())[:-4]))
    print('-------------------------------------------')
    model = word2vec.Word2Vec(corpus, sg=0,
                              workers=mp.cpu_count(),
                              compute_loss=True,
                              callbacks=[GensimCallback(verbose_round=3)],
                              **kwargs)
    print('-------------------------------------------')
    print('[INFO] CBOW embedding end at {}\n'.format(
        str(datetime.now())[:-4]))

    # 保存Embedding模型
    # ---------------------------
    file_processor = LoadSave(
        dir_name='./pretraining_models/', verbose=1)

    if is_save_model:
        file_processor.save_data(
            file_name='{}.pkl'.format(model_name),
            data_file=model)
    return model


def compute_tfidf_feats(corpus=None, max_feats=100, ngram_range=None):
    '''计算稀疏形式的TF-IDF特征'''
    if ngram_range is None:
        ngram_range = (1, 1)

    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm='l2',
                                 max_features=max_feats, max_df=1.0,
                                 analyzer='word', ngram_range=ngram_range,
                                 token_pattern=r'(?u)\b\w+\b')
    tfidf_array = vectorizer.fit_transform(corpus)

    return tfidf_array, vectorizer


def compute_embedding(corpus, word2vec, embedding_size):
    '''将句子转化为embedding vector'''
    embedding_mat = np.zeros((len(corpus), embedding_size))

    for ind, seq in enumerate(corpus):
        seq_vec, word_count = np.zeros((embedding_size, )), 0
        for word in seq:
            if word in word2vec:
                seq_vec += word2vec[word]
                word_count += 1

            if word_count != 0:
                embedding_mat[ind, :] = seq_vec / word_count
    return embedding_mat


if __name__ == '__main__':
    CBOW_MODEL_NAME = 'cbow_model'
    EMBEDDING_DIM = 128

    # 读入原始的训练与测试数据
    # -------------------------
    file_processor = LoadSave(dir_name='./cached_data/')
    total_targid_list = file_processor.load_data(
        file_name='total_targid_list.pkl')
    total_timestamp_list = file_processor.load_data(
        file_name='total_timestamp_list.pkl')

    total_feat_mat = None

    # 读入原始的训练与测试数据
    # -------------------------
    tmp_feat_list = list(map(comput_list_stat_feats, total_timestamp_list))
    tmp_feat_array = np.vstack(tmp_feat_list)
    tmp_feat_array = tmp_feat_array / 1000 / 3600

    if total_feat_mat is None:
        total_feat_mat = csr_matrix(tmp_feat_array)

    # targid特征抽取部分
    # -------------------------

    # TF-IDF
    # *****************
    for i in range(len(total_targid_list)):
        total_targid_list[i] = [str(item) for item in total_targid_list[i]]
        total_targid_list[i] = ' '.join(total_targid_list[i])
    tmp_feat_sp_array, encoder = compute_tfidf_feats(
        total_targid_list, max_feats=512)
    total_feat_mat = hstack([total_feat_mat, tmp_feat_sp_array]).tocsr()

    # word2vec embedding
    # *****************
    for i in range(len(total_targid_list)):
        total_targid_list[i] = total_targid_list[i].split(' ')

    if CBOW_MODEL_NAME:
        file_processor = LoadSave(dir_name='./pretraining_models/')
        cbow_model = file_processor.load_data(file_name=CBOW_MODEL_NAME+'.pkl')
    else:
        cbow_model = compute_cbow_embedding(
            corpus=total_targid_list, negative=20,
            min_count=2, window=128,
            vector_size=EMBEDDING_DIM, epochs=30)

    # 计算句子向量
    cbow_embedding_mat = compute_embedding(
        total_targid_list, cbow_model.wv, EMBEDDING_DIM)
    cbow_embedding_mat = csr_matrix(cbow_embedding_mat)

    # Spare matrix
    total_feat_mat = hstack([total_feat_mat, cbow_embedding_mat]).tocsr()

    # 存储Embedding特征工程结果
    # -------------------------
    file_processor.save_data(
        file_name='total_sp_embedding_mat.pkl', data_file=total_feat_mat)
