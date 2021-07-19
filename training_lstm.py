#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107191059
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(training_lstm.py)借助预训练的词向量，采用LSTM训练模型。
'''

import gc
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (LSTM, GRU, BatchNormalization, Bidirectional,
                                     Dense, Dot, Dropout, Embedding,
                                     GlobalAveragePooling1D,
                                     GlobalMaxPooling1D, Input, Lambda,
                                     LayerNormalization, Permute,
                                     SpatialDropout1D, concatenate, multiply,
                                     subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from dingtalk_remote_monitor import RemoteMonitorDingTalk, send_msg_to_dingtalk
from utils import GensimCallback, LoadSave

GLOBAL_RANDOM_SEED = 1995
np.random.seed(GLOBAL_RANDOM_SEED)
tf.random.set_seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings("ignore")

TASK_NAME = 'iflytek_commodity_recommendation_2021'
GPU_ID = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 限制Tensorflow只使用GPU ID编号的GPU
        tf.config.experimental.set_visible_devices(gpus[GPU_ID], 'GPU')

        # 限制Tensorflow不占用所有显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)
###############################################################################

def build_embedding_matrix(word2idx=None, word2embedding=None,
                           max_vocab_size=300, embedding_size=128,
                           oov_token=None, verbose=False):
    """利用idx2embedding，组合重新编码过的word2idx。

    @Parameters:
    ----------
    word2idx: {dict-like}
        将词语映射到index的字典。键为词语，值为词语对应的index。
    word2embedding: {array-like or dict-like}
        可按照Index被索引的对象，idx2embedding对应了词语的向量，
        通常是gensim的模型对象。
    embedding_size: {int-like}
        embedding向量的维度。
    max_vocab_size: {int-like}
        词表大小，index大于max_vocab_size的词被认为是OOV词。
    oov_token: {str-like}
        未登录词的Token表示。
    verbose: {bool-like}
        是否打印tqdm信息。

    @Return:
    ----------
    embedding_mat: {array-like}
        可根据index在embedding_mat的行上进行索引，获取词向量

    @References:
    ----------
    [1] https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold
    [2] https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471
    """
    if word2idx is None or word2embedding is None:
        raise ValueError("Invalid Input Parameters !")
    embedding_mat = np.zeros((max_vocab_size+1, embedding_size))

    for word, idx in tqdm(word2idx.items(), disable=not verbose):
        if idx > max_vocab_size:
            continue

        if word in word2embedding:
            embedding_vec = word2embedding[word]
        else:
            embedding_vec = np.array([1] * embedding_size)

        embedding_mat[idx] = embedding_vec
    return embedding_mat


def build_embedding_sequence(train_corpus=None, test_corpus=None,
                             max_vocab_size=1024,
                             max_sequence_length=128,
                             word2embedding=None,
                             oov_token="UNK"):
    """利用训练与测试语料，基于embedding_model构建用于神经网络的embedding矩阵。

    @Parameters:
    ----------
    train_corpus: {list-like}
        包含训练样本的文本序列。每一个元素为一个list，每一个list为训练集的一条句子。
    test_corpus: {list-like}
        包含测试样本的文本序列。每一个元素为一个list，每一个list为测试集的一条句子。
    max_vocab_size: {int-like}
        仅仅编码词频最大的前max_vocab_size个词汇。
    max_sequence_length: {int-like}
        将每一个句子padding到max_sequence_length长度。
    word2embedding: {indexable object}
        可索引的对象，键为词，值为embedding向量。
    oov_token: {str-like}
        语料中的oov_token。

    @Returen:
    ----------
    train_corpus_encoded: {list-like}
        经过编码与补长之后的训练集语料数据。
    test_corpus_encoded: {list-like}
        经过编码与补长之后的测试集语料数据。
    embedding_meta: {dict-like}
        包含embedding_mat的基础信息的字典。
    """
    try:
        embedding_size = word2embedding["feat_dim"]
    except KeyError:
        embedding_size = word2embedding.layer1_size

    # 拼接train与test语料数据，获取总语料
    # --------------------------------
    total_corpus = train_corpus + test_corpus

    # 序列化编码语料数据
    # --------------------------------
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(total_corpus)

    word2idx = tokenizer.word_index
    train_corpus_encoded = tokenizer.texts_to_sequences(train_corpus)
    test_corpus_encoded = tokenizer.texts_to_sequences(test_corpus)

    # 补长训练与测试数据，默认以0进行填补
    train_corpus_encoded = pad_sequences(
        train_corpus_encoded, maxlen=max_sequence_length)
    test_corpus_encoded = pad_sequences(
        test_corpus_encoded, maxlen=max_sequence_length)

    # 构造预训练的embedding matrix
    # --------------------------------
    embedding_mat = build_embedding_matrix(
        word2idx=word2idx,
        word2embedding=word2embedding,
        max_vocab_size=max_vocab_size,
        embedding_size=embedding_size,
        oov_token=oov_token)

    embedding_meta = {}
    embedding_meta["embedding_size"] = embedding_mat.shape[1]
    embedding_meta["max_len"] = max_sequence_length
    embedding_meta["max_vocab"] = max_vocab_size
    embedding_meta["embedding_mat"] = embedding_mat
    embedding_meta["tokenizer"] = tokenizer

    return train_corpus_encoded, test_corpus_encoded, embedding_meta


if __name__ == '__main__':
    pass



