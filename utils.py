# coding: UTF-8
import os
import re

import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config):

    def load_dataset(path, pad_size=128):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                all_element = re.split(r'\s+', lin)
                content, label = all_element[0: -1], all_element[-1]
                content = list(map(float, content))  # 列表中的每个元素转化为 int 类型
                seq_len = len(content)
                if pad_size:
                    if len(content) < pad_size:
                        content.extend([PAD] * (pad_size - len(content)))
                    else:
                        content = content[:pad_size]
                        seq_len = pad_size
                contents.append((content, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        x = torch.Tensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)


def generate_dataset():
    current_path = os.getcwd()
    os.makedirs(current_path + '/dataset/data', exist_ok=True)
    os.makedirs(current_path + '/dataset/saved_dict', exist_ok=True)

    data = pd.read_excel(current_path +'/dataset/10-600-4类.xlsx', header=None)  # 读取 excel 中的数据

    X = data.iloc[:, 1:-1]  # 取出特征赋给 X
    y = data.iloc[:, -1]  # 取出标签赋给 y
    y = y - 1  # 标签从0开始，所以要对每个元素-1

    class_list = y.unique()  # 统计类别标签

    with open(current_path + '/dataset/data/class.txt', 'w') as f:
        for item in class_list:
            f.write('%s\n' % item)

    # X = torch.from_numpy(X.values)  # 转换成 tensor
    # y = torch.from_numpy(y.values)  # 转换成 tensor

    # 训练集和测试集划分
    X_train, X_res, y_train, y_res = train_test_split(X, y, test_size=0.4, random_state=2023)
    X_valid, X_test, y_valid, y_test = train_test_split(X_res, y_res, test_size=0.5, random_state=2023)

    train_data = pd.concat([X_train, y_train], axis=1)
    train_output_file = '/dataset/data/train.txt'
    with open(current_path + train_output_file, 'w') as f:
        f.write(train_data.to_string(index=False, header=False))

    valid_data = pd.concat([X_valid, y_valid], axis=1)
    valid_output_file = '/dataset/data/valid.txt'
    with open(current_path + valid_output_file, 'w') as f:
        f.write(valid_data.to_string(index=False, header=False))

    test_data = pd.concat([X_test, y_test], axis=1)
    test_output_file = '/dataset/data/test.txt'
    with open(current_path + test_output_file, 'w') as f:
        f.write(test_data.to_string(index=False, header=False))

    # # joblib.dump((X_train, X_valid, X_test, y_train, y_valid, y_test), 'dataset.joblib')  # 将划分好的数据集存储到 dataset.joblib
    # joblib.dump((X_train, y_train), 'classifier/dataset/data/train.joblib')
    # joblib.dump((X_valid, y_valid), 'classifier/dataset/data/dev.joblib')
    # joblib.dump((X_test, y_test), 'classifier/dataset/data/test.joblib')

    # 读取时使用以下函数
    # X_train, X_test, y_train, y_test = joblib.load('dataset.joblib')
